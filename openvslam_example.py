import numpy as np
import openvslam
import os
import cv2 as cv
import numpy as np

np.set_printoptions(np.inf)

config = openvslam.config(config_file_path="./equirectangular.yaml")

voc ="./orb_vocab.fbow"
slam = openvslam.OpenVSLAM(config,voc)
image_file = r"./imags"
file_list = os.listdir(image_file)
file_list.sort()

for timestamp,frame_name in enumerate(file_list):
    frame = cv.imread(os.path.join(image_file,frame_name))
    ok, pose = slam.track(frame, timestamp,np.array([]))
    if ok == True:
        id,cloud = slam.get_sparse_cloud()
        print(np.array(cloud).shape)
slam.shutdown()
