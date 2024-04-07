from scipy.spatial.transform import Rotation as R
import numpy as np
import os

# path to TartanAir images and ground truth poses
images = 'Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/image_left/'
poses = 'Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/pose_left.txt'

# path to store output txt file to use with SuperGlue
output = 'Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/pose_left_SuperGlue.txt'

# get list of image names
image_names = sorted(os.listdir(images))

# read TartanAir poses
with open(poses, 'r') as f:
    lines = f.readlines()

# initialise list to store SuperGlue poses
super_glue_poses = []

# camera intrinsics
K0 = K1 = np.array([320.0, 0., 320.0, 0., 320.0, 240.0, 0., 0., 1.])

# iterate over pairs of consecutive images
for i in range(len(lines) - 1):
    # parse TartanAir poses
    pose0 = np.array(lines[i].split(), dtype=float)
    pose1 = np.array(lines[i+1].split(), dtype=float)

    # convert quaternions to rotation matrices
    rot0 = R.from_quat(pose0[3:7]).as_matrix()
    rot1 = R.from_quat(pose1[3:7]).as_matrix()

    # get translation vectors
    t0 = pose0[:3]
    t1 = pose1[:3]

    # coordinate transformation from NED to ENU
    transform = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])

    # apply transformation to rotation
    rot0 = transform @ rot0 @ transform.T
    rot1 = transform @ rot1 @ transform.T

    # apply transformation to translation
    t0 = transform @ t0
    t1 = transform @ t1

    # form 4x4 transformation matrices
    T0 = np.eye(4)
    T0[:3, :3] = rot0
    T0[:3, 3] = t0

    T1 = np.eye(4)
    T1[:3, :3] = rot1
    T1[:3, 3] = t1

    # compute relative pose
    T = np.linalg.inv(T0) @ T1

    # get image names
    img0_name = image_names[i]
    img1_name = image_names[i+1]

    # form SuperGlue line
    super_glue_line = [img0_name, img1_name, '0', '0'] + K0.tolist() + K1.tolist() + T.flatten().tolist()

    # append to list
    super_glue_poses.append(super_glue_line)

# delete the old file if it exists
if os.path.exists(output):
    os.remove(output)

# create a new output file
if not os.path.exists(output):
    with open(output, 'w') as f:
        pass

# write SuperGlue poses to file
with open(output, 'w') as f:
    for pose in super_glue_poses:
        f.write(' '.join(map(str, pose)) + '\n')

# python3 match_pairs.py --input_pairs ~/ros_ws/src/Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/pose_left_SuperGlue.txt --input_dir ~/ros_ws/src/Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/image_left/ --output_dir ~/ros_ws/src/Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/SuperGlue_matches/ --eval --viz --show_keypoints --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float
# python3 match_pairs.py --input_pairs ~/ros_ws/src/Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P002/pose_left_SuperGlue.txt --input_dir ~/ros_ws/src/Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P002/image_left/ --output_dir ~/ros_ws/src/Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P002/SuperGlue_matches/ --superglue outdoor --eval --viz --show_keypoints
