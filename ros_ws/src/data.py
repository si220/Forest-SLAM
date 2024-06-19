import torch
import rosbag
import rospy
from cv_bridge import CvBridge
import tf.transformations as tf_trans
import cv2
import numpy as np
import os
import shutil
from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

bridge = CvBridge()

bag = rosbag.Bag('Datasets/BotanicGarden/1006_01_img10hz600p.bag')

data_dir = 'Datasets/BotanicGarden/training_data/'

# delete existing data folder
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

# create new data folder
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# extrinsic params for the transformation from VLP16 to RGB0 coordinate frame
T_rgb0_vlp16 = np.array([[0.0238743541600432, -0.999707744440396, 0.00360642510766516, 0.138922870923538],
                          [-0.00736968896588375, -0.00378431903190059, -0.999965147452649, -0.177101909101325],
                          [0.999687515506770, 0.0238486947027063, -0.00745791352160211, -0.126685267545513],
                          [0.0, 0.0, 0.0, 1.0]])

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

# instantiate the SuperPoint and SuperGlue models
feature_matcher = Matching(model_config).to(device)

# create dictionaries to store the ground truth poses from each bag
gt_poses = {}

# loop through ground truth poses in the rosbag
for topic, msg, t in bag.read_messages(topics=['/gt_poses']):
    gt_poses[t.to_sec()] = msg.pose

# function to find current pose based on closest timestamp
def find_closest_timestamp(target, timestamp_dict):
    # get the list of timestamps
    timestamps = np.array(list(timestamp_dict.keys()))

    # find the index of the closest timestamp
    idx = np.argmin(np.abs(timestamps - target))

    # return the pose corresponding to the closest timestamp
    return timestamp_dict[timestamps[idx]]

# initialise previous image and relative pose
prev_img = None
prev_tf_mat = None
index = 0

# initialise trajectory
cumulative_gt_tf_mat = np.eye(4)

for topic, msg, t in bag.read_messages(topics=['/dalsa_rgb/left/image_raw']):
    # exit on ctrl + c
    if rospy.is_shutdown():
        break

    # convert the image to greyscale and normalise it
    cur_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    cur_img = torch.from_numpy(cur_img/255.).float()[None, None].to(device)

    # get the corresponding ground truth pose
    cur_pose = find_closest_timestamp(t.to_sec(), gt_poses)
    cur_quat = [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]
    cur_tf_mat = tf_trans.quaternion_matrix(cur_quat)
    cur_tf_mat[:3, 3] = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]

    # transform from VLP16 LiDAR coordinate frame to dalsa_rgb/left coordinate frame
    cur_tf_mat = np.dot(T_rgb0_vlp16, cur_tf_mat)

    # check if there are 2 frames to compare
    if prev_img is not None and prev_tf_mat is not None:
        # perform the matching
        pred = feature_matcher({'image0': prev_img, 'image1': cur_img})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # only keep the matched keypoints
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # get ground truth relative and cumulative poses
        relative_tf_mat = np.dot(np.linalg.inv(prev_tf_mat), cur_tf_mat)
        cumulative_tf_mat = np.dot(cumulative_gt_tf_mat, relative_tf_mat)

        # save the data to files
        data = {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf,
            'rel_pose': relative_tf_mat,
            'abs_pose': cumulative_tf_mat
        }

        np.save(os.path.join(data_dir, f'data_{index}.npy'), data, allow_pickle=True)
        print(f'index: {index}')

    # update the previous image and relative pose
    prev_img = cur_img
    prev_tf_mat = cur_tf_mat
    index += 1
