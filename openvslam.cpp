
#include<opencv2/core/core.hpp>
#include<Python.h>
#include<pybind11/stl.h>
#include<pybind11/pybind11.h>
#include<pybind11/eigen.h>
#include<stella_vslam/system.h>
#include<stella_vslam/type.h>
#include<stella_vslam/data/keyframe.h>
#include<stella_vslam/data/landmark.h>
#include<stella_vslam/publish/map_publisher.h>
#include<stella_vslam/publish/frame_publisher.h>
#include<yaml-cpp/yaml.h>
#include<future>
#include<iostream>
#include<opencv2/core/core.hpp>
#include<yaml-cpp/yaml.h>




using namespace std;
#define Py_LIMITED_API 1

#if CV_MAJOR_VERSION < 4
// OpenCV 4 adopts AccessFlag type instead of int
typedef int AccessFlag;
#endif

namespace py = pybind11;

class NDArrayConverter {
public:
    static bool init_numpy();   // must call this first, or the other routines don't work!
    static bool toMat(PyObject* o, cv::Mat &m);
    static PyObject* toNDArray(const cv::Mat& mat);
};

namespace pybind11{namespace detail{
template <> struct type_caster<cv::Mat>{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));
    bool load(handle src, bool){
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval){
        return handle(NDArrayConverter::toNDArray(m));
    }
};
}} // namespace pybind11::detail

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#if PY_VERSION_HEX >= 0x03000000
    #define PyInt_Check PyLong_Check
    #define PyInt_AsLong PyLong_AsLong
#endif

struct Tmp {
    const char * name;
    Tmp(const char * name):name(name){}
} info("return value");

bool NDArrayConverter::init_numpy(){
    // this has to be in this file, since PyArray_API is defined as static
    import_array1(false);
    return true;
}

/*
 * The following conversion functions are taken/adapted from OpenCV's cv2.cpp file
 * inside modules/python/src2 folder (OpenCV 3.1.0)
 */
static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...){
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads(){
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL(){
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try {PyAllowThreads allowThreads; expr;} \
catch (const cv::Exception &e){PyErr_SetString(opencv_error, e.what()); return 0;}

using namespace cv;

class NumpyAllocator : public MatAllocator{
public:
    NumpyAllocator(){stdAllocator = Mat::getStdAllocator();}
    ~NumpyAllocator(){}

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const{
        UMatData* u = new UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( int i = 0; i < dims - 1; i++ )
            step[i] = (size_t)_strides[i];
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, AccessFlag flags, UMatUsageFlags usageFlags) const {
        if( data != 0 ){
            CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
            _sizes[dims++] = cn;
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        return allocate(o, dims0, sizes, type, step);
    }

    bool allocate(UMatData* u, AccessFlag accessFlags, UMatUsageFlags usageFlags) const {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const{
        if(!u)
            return;
        PyEnsureGIL gil;
        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);
        if(u->refcount == 0){
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};

NumpyAllocator g_numpyAllocator;

bool NDArrayConverter::toMat(PyObject *o, Mat &m){
    bool allowND = true;
    if(!o || o == Py_None){
        if(!m.data)
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if(PyInt_Check(o)){
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)o)), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if(PyFloat_Check(o)){
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if(PyTuple_Check(o)){
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = Mat(sz, 1, CV_64F);
        for(i = 0; i < sz; i++){
            PyObject* oi = PyTuple_GET_ITEM(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else{
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    if(!PyArray_Check(o)){
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if(type < 0){
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG ){
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        } else {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM){
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for(int i = ndims-1; i >= 0 && !needcopy; i--)
        if(
            (i == ndims-1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize)     ||
            (i  < ndims-1 && _sizes[i] > 1 &&         _strides[i] < _strides[i+1]) 
        )
            needcopy = true;

    if(ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2])
        needcopy = true;

    if(needcopy){
        if(needcast){
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        } else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }
        _strides = PyArray_STRIDES(oarr);
    }

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for(int i = ndims - 1; i >= 0; --i){
        size[i] = (int)_sizes[i];
        if ( size[i] > 1 ){
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        } else {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // handle degenerate case
    if(ndims == 0){
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if(ismultichannel){
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if(ndims > 2 && !allowND){
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    m.addref();

    if(!needcopy){
        Py_INCREF(o);
    }
    m.allocator = &g_numpyAllocator;

    return true;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m){
    if( !m.data ){
        Py_RETURN_NONE;
    }
    Mat temp, *p = (Mat*)&m;
    if(!p->u || p->allocator != &g_numpyAllocator){
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    PyObject* o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}








using namespace stella_vslam;
class OpenVSLAM
{
public:
    using Mat44_t = Eigen::Matrix4d;
    using Ptr = std::shared_ptr<OpenVSLAM>;
    OpenVSLAM(const std::shared_ptr<stella_vslam::config> config_set,const string vocab_file_path):
        m_base_point_id(0),
        m_max_point_id(0),
        m_nrof_keyframes(0),
        m_previous_state(stella_vslam::tracker_state_t::Initializing),
        //m_max_keyframe_links(10),
        m_last_keyframe(nullptr) 
    {
        // m_config = config_set;
        m_vslam =  std::make_shared<stella_vslam::system>(config_set, vocab_file_path);
        m_frame_publisher = m_vslam->get_frame_publisher();
        m_map_publisher = m_vslam->get_map_publisher();
        m_vslam->startup(true);
    }; 

    void shutdown()
    {
        m_vslam->shutdown();
    };
    void reset(){
     m_vslam->request_reset();
    };
    std::pair<bool, Mat44_t> track(const cv::Mat& image,const double timestamp,const cv::Mat& Mask = cv::Mat{}){
        
        std::shared_ptr<stella_vslam::Mat44_t> T_w2c;
        T_w2c = m_vslam->feed_monocular_frame(image, timestamp);
        // m_mutex_last_drawn_frame.lock();
        // m_last_drawn_frame = m_frame_publisher->draw_frame();
        // m_mutex_last_drawn_frame.unlock();
        stella_vslam::tracker_state_t tracker_state ;
        std::string state_str = m_frame_publisher->get_tracking_state();
        if (state_str ==  "Tracking")
            tracker_state = stella_vslam::tracker_state_t::Tracking;
        else if (state_str =="Lost")
        {
            tracker_state = stella_vslam::tracker_state_t::Lost;
        }
        else if (state_str =="Initializing")
        {
            tracker_state = stella_vslam::tracker_state_t::Initializing;
        }
        else
            throw std::runtime_error("unknown state");


        if (tracker_state == stella_vslam::tracker_state_t::Tracking)
        {
            if (T_w2c == nullptr)
                return ptr2pose(T_w2c);
             //m_last_drawn_frame = m_frame_publisher->draw_frame();
            std::cout<<"current tracking state is Tracking"<<endl;
            std::vector<std::shared_ptr<stella_vslam::data::keyframe>>  keyframes;
            unsigned int current_nrof_keyframes = m_map_publisher->get_keyframes(keyframes);
            m_mutex_last_keyframe.lock();
            if (m_last_keyframe == nullptr)
                m_last_keyframe = keyframes.back().get();
            else
            {
                for (auto kf : keyframes)
                    if (kf->id_ > m_last_keyframe->id_)
                        m_last_keyframe = kf.get();
            }
            m_mutex_last_keyframe.unlock();

            m_sparse_cloud = getTrackedMapPoints();
            
            m_previous_state = tracker_state;
            if (m_nrof_keyframes == 0 && current_nrof_keyframes > 0){
                m_nrof_keyframes = current_nrof_keyframes;
                return ptr2pose(T_w2c);
            }
            else if (current_nrof_keyframes != m_nrof_keyframes)
            {
                m_last_keyframe->set_not_to_be_erased();
                m_nrof_keyframes = current_nrof_keyframes;
                return ptr2pose(T_w2c);
            }
            else{
                return ptr2pose(T_w2c);
            }
            }
        else if    ((m_previous_state == stella_vslam::tracker_state_t::Tracking || m_previous_state == stella_vslam::tracker_state_t::Lost) &&
            (tracker_state == stella_vslam::tracker_state_t::Initializing)){
                internalReset();
                //m_reset_callback();
                return ptr2pose(T_w2c);
            }     
        m_previous_state = tracker_state;
        return ptr2pose(T_w2c);

    };
    std::pair<std::vector<uint32_t>, cv::Mat> get_sparse_cloud(){
    return m_sparse_cloud;
    };
    std::pair<std::vector<uint32_t>, cv::Mat> getTrackedMapPoints(){
        m_mutex_last_keyframe.lock();
        vector<std::shared_ptr<stella_vslam::data::landmark>> landmarks = m_last_keyframe->get_landmarks();
        m_mutex_last_keyframe.unlock();
        std::vector<uint32_t> point_ids;
        cv::Mat points;
        points.reserve(landmarks.size());
        for (const auto &lm : landmarks)
        {
            if (!lm || lm->will_be_erased())
            {
            continue;
            }
            stella_vslam::Vec3_t pos = lm->get_pos_in_world();
            cv::Mat pt = (cv::Mat_<double>(1, 3) << pos[0], pos[1], pos[2]);
            points.push_back(pt);
            uint32_t point_id = extractPointId(lm.get());
            point_ids.push_back(point_id);
            if (point_id > m_max_point_id)
                m_max_point_id = point_id;
        }

        return std::pair<std::vector<uint32_t>,cv::Mat>(point_ids, points);
 
        
    };
    uint32_t extractPointId(stella_vslam::data::landmark* lm){
        return m_base_point_id + lm->id_;
    };
    std::pair<bool, Mat44_t> ptr2pose(std::shared_ptr<Mat44_t> pose_ptr)
    {
        if(pose_ptr == nullptr)
            // No pose
            return std::pair<bool, Mat44_t>(false, Mat44_t());
        else
            // Pose ok
            return std::pair<bool, Mat44_t>(true, (Mat44_t) *pose_ptr);
      };
    void internalReset()
    {
        std::lock_guard<std::mutex> lock(m_mutex_last_keyframe);
        m_last_keyframe = nullptr;
        m_nrof_keyframes = 0;
        // The new base point id is the maximum point id ever recognised +1. This way even though the SLAM starts counting
        // at 0 again, we still have a unique id for all the points.
        m_base_point_id = m_max_point_id+1;
        //m_keyframe_links.clear();
    };
private:
    uint32_t m_base_point_id;
    uint32_t m_max_point_id;
    unsigned int m_nrof_keyframes;
    cv::Mat m_last_drawn_frame;
    mutable std::mutex m_mutex_last_drawn_frame;
    mutable std::mutex m_mutex_last_keyframe;
    stella_vslam::data::keyframe* m_last_keyframe;
    stella_vslam::tracker_state_t m_previous_state;
    std::shared_ptr<stella_vslam::system> m_vslam;
    std::shared_ptr<stella_vslam::publish::frame_publisher> m_frame_publisher;
    std::shared_ptr<stella_vslam::publish::map_publisher> m_map_publisher;
    std::future<void> m_future_update_keyframes;
    std::pair<std::vector<uint32_t>, cv::Mat> m_sparse_cloud;
};






PYBIND11_MODULE(openvslam, m){
    NDArrayConverter::init_numpy();
    py::class_<config, std::shared_ptr<config>>(m, "config")
        .def(py::init<const std::string&>(), py::arg("config_file_path"))
        .def(py::init<const YAML::Node&, const std::string&>(), py::arg("yaml_node"), py::arg("config_file_path") = "")
        .def_readonly("yaml_node_", &config::yaml_node_)
        ;
    py::class_<OpenVSLAM>(m,"OpenVSLAM")
        .def(py::init<const std::shared_ptr<config>&, const std::string&>(), py::arg("cfg"), py::arg("vocab_file_path"))
        .def("shutdown",&OpenVSLAM::shutdown)
        .def("track", &OpenVSLAM::track)
        // &self, const cv::Mat &image, const double timestamp, const cv::Mat &Mask) {
        //         return self.track(image, timestamp, sMak);
        //     },
        //     py::arg("image"), py::arg("timestamp")=0.0, py::arg("Mask") = cv::Mat{})
        .def("getTrackedMapPoints",&OpenVSLAM::getTrackedMapPoints)
        .def("get_sparse_cloud",&OpenVSLAM::get_sparse_cloud)
        // .def("getTrackedMapPoints",[](OpenVSLAM::getTrackedMapPoints &self) {
        //     return self.getTrackedMapPoints();
        // })
        ;
}       











