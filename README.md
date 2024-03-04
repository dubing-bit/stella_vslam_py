# Use python bindings stella-vslam 
Get camera poses and Landmars (or map points) using python bindings to Stella-VSLAM. Most of the code references (copies) [AlejandroSilvestri's](https://github.com/AlejandroSilvestri/Stella-VSLAM-Python-bindings) project, with the difference of implementing the landmarks calling functionality.
## install
- install [stella-vslam](https://stella-cv.readthedocs.io/en/latest/installation.html).
- Install python3, pybind11 and other libraries（i forget）.
- Execute the following command in the terminal
`g++ -O3 -Wall -shared -std=c++14 -fPIC $(python3-config --includes) -I/usr/local/include/stella_vslam/3rd/json/include -I/usr/local/include/eigen3 -I/usr/local/include/opencv4 ./openvslam.cpp -o stellavslam$(python3-config --extension-suffix) -lstella_vslam`
## how to use
reference `openvslam_example.py`
