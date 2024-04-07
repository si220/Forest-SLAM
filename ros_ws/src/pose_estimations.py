import matplotlib
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import estimate_pose
matplotlib.use('TkAgg')

# load the pose data from the file
pose_file = 'Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/pose_left.txt'
data = np.loadtxt(pose_file)

# extract the position data
positions_ned = data[:, :3]

# subtract the initial position from all positions to make them start from the origin
positions_ned -= positions_ned[0]

# convert NED coordinate frame to ENU
positions_enu_gt = np.zeros_like(positions_ned)
positions_enu_gt[:, 0] = positions_ned[:, 1]  # East (E) from North (N)
positions_enu_gt[:, 1] = positions_ned[:, 0]  # North (N) from East (E)
positions_enu_gt[:, 2] = -positions_ned[:, 2]  # Up (U) from Down (D)

# input dir
image_dir = 'Datasets/TartanAir/seasonsforest/image_left/seasonsforest/Easy/P001/image_left/'

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# disable gradient calculations for faster inference
torch.set_grad_enabled(False)

K0 = K1 = np.array([[320.0, 0., 320.0],
               [0., 320.0, 240.0],
               [0., 0., 1.]])

# SuperPoint and SuperGlue hyperparams
model_config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

# initialise SuperPoint and SuperGlue
network = Matching(model_config).eval().to(device)

# list of image file names
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

# initialise list to store poses
poses = []

# iterate over pairs of images
for i in range(len(image_files) - 1):
    print(i)
    
    img1_path = os.path.join(image_dir, image_files[i])
    img2_path = os.path.join(image_dir, image_files[i + 1])

    # load images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # convert to tensors
    img1 = torch.from_numpy(img1/255.).float()[None, None].to(device)
    img2 = torch.from_numpy(img2/255.).float()[None, None].to(device)

    # perform forward pass to get predictions
    model_pred = network({'image0': img1, 'image1': img2})
    model_pred = {j: k[0].cpu().numpy() for j, k in model_pred.items()}

    # get SuperPoint keypoints
    keypoints_0, keypoints_1 = model_pred['keypoints0'], model_pred['keypoints1']

    # get number of matches and match confidence from SuperGlue
    num_matches, match_confidence = model_pred['matches0'], model_pred['matching_scores0']

    # only keep features that have been matched
    valid = num_matches > -1
    matched_keypoints_0 = keypoints_0[valid]
    matched_keypoints_1 = keypoints_1[num_matches[valid]]

    # estimate relative pose
    relative_pose = estimate_pose(matched_keypoints_0, matched_keypoints_1, K0, K1, 1.)

    # add to poses list if a valid pose was estimated
    if relative_pose is not None:
        poses.append(relative_pose)

# function to create a 4x4 transformation matrix from rotation and translation
def create_transformation_matrix(R, t):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    return transformation_matrix

# function to chain relative poses and create a trajectory
def create_trajectory(relative_poses):
    # initialise trajectory with a 4x4 identity matrix
    trajectory = [np.eye(4)]

    # chain the relative poses
    for pose_tuple in relative_poses:
        if pose_tuple is not None:
            R, t, inliers = pose_tuple
            transformation_matrix = create_transformation_matrix(R, t)
            trajectory.append(trajectory[-1] @ transformation_matrix)

    return trajectory

# create the trajectory
trajectory = create_trajectory(poses)

# function to extract positions from 4x4 transformation matrices
def extract_positions(trajectory):
    positions = np.array([pose[:3, 3] for pose in trajectory])
    return positions

# extract positions from the trajectory
positions_enu_est = extract_positions(trajectory)

# compute the scale factor between the estimated and ground truth trajectories
scale_factor = np.linalg.norm(positions_enu_gt[1] - positions_enu_gt[0]) / np.linalg.norm(positions_enu_est[1] - positions_enu_est[0])

# correct the scale of the estimated trajectory
positions_enu_est_corrected = positions_enu_est * scale_factor

# define the orders of axes to try
orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

# create a figure with subplots
fig = plt.figure(figsize=(18, 12))

for i, order in enumerate(orders, start=1):
    for j, order_gt in enumerate(orders, start=1):
        # create a 3D subplot
        ax = fig.add_subplot(len(orders), len(orders), (i-1)*len(orders)+j, projection='3d')

        # plot the ground truth trajectory with the current order of axes
        ax.plot(positions_ned[:, order_gt[0]], positions_ned[:, order_gt[1]], positions_ned[:, order_gt[2]], marker='o')

        # plot the estimated trajectory with the current order of axes
        ax.plot(positions_enu_est_corrected[:, order[0]], positions_enu_est_corrected[:, order[1]], positions_enu_est_corrected[:, order[2]], marker='o')

        # set labels for axes
        ax.set_xlabel('Axis ' + str(order[0]))
        ax.set_ylabel('Axis ' + str(order[1]))
        ax.set_zlabel('Axis ' + str(order[2]))

# show the plots
plt.show()
