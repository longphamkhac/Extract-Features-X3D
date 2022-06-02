import numpy as np
import os
import glob
from os import walk
from scipy.io import loadmat

# with open("E:\\RTFM\\test_X3D.txt") as f:
#     dirs = f.readlines()

rgb_list_file = "F:\\RTFM\\test_X3D.txt"
temporal_root = "F:\\RTFM\\test_frame_mask"

gt_files = os.listdir(temporal_root) # "F:\\RTFM\\test_frame_mask"
file_list = list(open(rgb_list_file)) # F:\\Features\\test_shanghai\\01_0015.npy

num_frames = 0
gt = []
index = 0
total = 0
abnormal_count = 0

for file in file_list:
    features = np.load(file.strip("\n"), allow_pickle = True)
    features = np.array(features, dtype = np.float32)

    num_frames = features.shape[0] * 16 # 432
    
    count = 0
    gt_file = file.strip("\n").split("\\")[-1] # 01_0015.npy
    gt_file = os.path.join(temporal_root, gt_file) # F:\RTFM\test_frame_mask\01_0015.npy
    
    if not os.path.isfile(gt_file): # Normal Video
        print("Normal Video:", str(file))
        for i in range(0, num_frames):
            gt.append(0)
            count += 1
    
    else: # Abnormal Video
        print("Abnormal Video:", str(file))
        abnormal_count += 1
        ground_annotation = np.load(gt_file) # (433,)
        ground_annotation = list(ground_annotation)

        if len(ground_annotation) < num_frames:
            last_frame_label = ground_annotation[-1]
            for i in range(len(ground_annotation), num_frames):
                ground_annotation.append(last_frame_label)
        else:
            ground_annotation = ground_annotation[:num_frames]
        
        if len(ground_annotation) != num_frames:
            print("Wrong Frame Number!")
            print("Wrong File: ", file)
            exit(1)
        
        count += len(ground_annotation)
        gt.extend(ground_annotation)
    
    index += 1
    total += count

print(len(gt))
np.save("GT_SH_X3D.npy", gt)
print("Save Successfully!!")