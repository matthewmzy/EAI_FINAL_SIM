import os
import numpy as np

train_path = 'assignment2_zzh/data/power_drill/train'

obj_trans_list = []
obj_rot_list = []
for data in os.listdir(train_path):
    obj_pose = os.path.join(train_path, data, 'object_pose.npy')
    cam_pose = os.path.join(train_path, data, 'camera_pose.npy')
    obj_pose = np.load(obj_pose)
    cam_pose = np.load(cam_pose)
    obj_pose = np.linalg.inv(cam_pose) @ obj_pose  # Convert to camera frame
    obj_trans = obj_pose[:3,3]
    obj_trans_list.append(obj_trans)
    obj_rot = obj_pose[:3,:3]
    obj_rot_list.append(obj_rot)

obj_trans_list = np.array(obj_trans_list)
obj_rot_list = np.array(obj_rot_list)

print(np.mean(obj_trans_list, axis=0))
print(np.mean(obj_rot_list, axis=0))