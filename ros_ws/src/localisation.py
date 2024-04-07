import matplotlib
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import estimate_pose
matplotlib.use('TkAgg')

# input dir
image_dir = 'Datasets/BotanicGarden/1018_dalsa_garden_short/1018_garden_short_imgs/00/c54d7a_png/'

# use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# disable gradient calculations for faster inference
torch.set_grad_enabled(False)

# camera intrinsic params from the given projection parameters
K0 = K1 = np.array([[642.9165664800531, 0., 460.1840658156501],
                    [0., 641.9171825800378, 308.5846449100310],
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

# create a 2D plot
fig, ax = plt.subplots()

# use interactive mode
plt.ion()
plt.show()

# initialise list to store poses
poses = []

# image dimensions for resizing images
img_dims = (960, 600)

# distortion coefficients
# k1, k2, p1, p2
dist_coeffs = np.array([-0.060164620903866, 0.094005180631043, 0, 0])

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

# function to extract positions from 4x4 transformation matrices
def extract_pos(trajectory):
    positions = np.array([pose[:3, 3] for pose in trajectory])
    return positions

# iterate over pairs of images
for i in range(len(image_files) - 1):
    print(i)

    # path to images
    img0_path = os.path.join(image_dir, image_files[i])
    img1_path = os.path.join(image_dir, image_files[i + 1])

    # load images
    img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

    # resize images
    img0 = cv2.resize(img0, img_dims)
    img1 = cv2.resize(img1, img_dims)

    # undistort images
    img0 = cv2.undistort(img0, K0, dist_coeffs)
    img1 = cv2.undistort(img1, K0, dist_coeffs)

    # convert to tensors
    img0 = torch.from_numpy(img0/255.).float()[None, None].to(device)
    img1 = torch.from_numpy(img1/255.).float()[None, None].to(device)

    # perform forward pass to get predictions
    model_pred = network({'image0': img0, 'image1': img1})
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

        # create the trajectory
        trajectory = create_trajectory(poses)

        # extract positions from the trajectory
        estimated_pos = extract_pos(trajectory)

        # clear the previous plot
        ax.clear()

        # plot the new trajectory
        ax.plot(estimated_pos[:, 0], estimated_pos[:, 1])

        # set the plot limits dynamically
        ax.set_xlim([estimated_pos[:, 0].min() - 1, estimated_pos[:, 0].max() + 1])
        ax.set_ylim([estimated_pos[:, 1].min() - 1, estimated_pos[:, 1].max() + 1])

        # draw the plot
        plt.draw()

        # pause before updating plot
        plt.pause(0.0001)

# save the final plot
plt.savefig('trajectory.png')
