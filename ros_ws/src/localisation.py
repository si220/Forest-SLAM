import torch
import rosbag
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import tf.transformations as tf_trans
import cv2
import numpy as np
import os
import shutil
from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# disable gradient calculations as not needed for inference
torch.set_grad_enabled(False)

bridge = CvBridge()

# initialise ros publisher
rospy.init_node('path_publisher')
gt_path_pub = rospy.Publisher('gt_trajectory', Path, queue_size=10)
est_path_pub = rospy.Publisher('est_trajectory', Path, queue_size=10)

# initialise an empty ground truth path message
gt_path = Path()
gt_path.header.frame_id = 'map'

# initialise an empty estimated path message
est_path = Path()
est_path.header.frame_id = 'map'

bag = rosbag.Bag('Datasets/BotanicGarden/1005_07_img10hz600p.bag')

# camera intrinsic params from the given projection parameters
K0 = K1 = np.array([[642.9165664800531, 0., 460.1840658156501],
                    [0., 641.9171825800378, 308.5846449100310],
                    [0., 0., 1.]])

# distortion params
k1 = -0.060164620903866
k2 = 0.094005180631043
p1 = 0.0
p2 = 0.0

dist_coeffs = np.array([k1, k2, p1, p2, 0])

# extrinsic params for the transformation from RGB0 to VLP16 coordinate frame
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

# initialise SuperPoint and SuperGlue
feature_matcher = Matching(model_config).eval().to(device)

# create a dictionary to store the ground truth poses
gt_poses = {}

# loop through ground truth poses in rosbag
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

# initialise previous image, timestamp, ground truth pose and transformation matrix
prev_img = None
prev_tf_mat = None
prev_translation = None

# initialise cumulative transformation matrices
cumulative_gt_tf_mat = np.eye(4)
cumulative_est_tf_mat = np.eye(4)

# initialise lists to store ground truth and estimated poses in TUM format
gt_poses_tum = []
est_poses_tum = []

# loop through image data in rosbag
for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw'])):
    # exit on ctrl + c
    if rospy.is_shutdown():
        break

    # convert the image to greyscale and normalise it
    cur_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cur_img = cv2.undistort(cur_img, K0, dist_coeffs)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    cur_img = torch.from_numpy(cur_img/255.).float()[None, None].to(device)

    # get the corresponding ground truth pose
    cur_pose = find_closest_timestamp(t.to_sec(), gt_poses)
    cur_quat = [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]
    cur_translation = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]
    cur_tf_mat = tf_trans.quaternion_matrix(cur_quat)
    cur_tf_mat[0:3, 3] = cur_translation
    cur_tf_mat = np.dot(T_rgb0_vlp16, cur_tf_mat)

    # select pairs that are 5 frames apart to ensure there is sufficient motion between frames
    if index % 5 == 0:
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

            # calculate the ground truth relative translation and rotation
            relative_gt_tf_mat = np.dot(np.linalg.inv(prev_tf_mat), cur_tf_mat)
            cumulative_gt_tf_mat = np.dot(cumulative_gt_tf_mat, relative_gt_tf_mat)

            # calculate the estimated relative translation and rotation
            E, mask = cv2.findEssentialMat(mkpts0, mkpts1, focal=K0[0,0], pp=(K0[0,2], K0[1,2]), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, rotation, translation, _ = cv2.recoverPose(E, mkpts0, mkpts1, focal=K0[0,0], pp=(K0[0,2], K0[1,2]))
            # scale = np.sqrt((cur_translation[0] - prev_translation[0])**2 +
            #                 (cur_translation[1] - prev_translation[1])**2 +
            #                 (cur_translation[2] - prev_translation[2])**2)
            # translation *= scale
            translation = translation.reshape(3)

            relative_est_tf_mat = np.eye(4)
            relative_est_tf_mat[:3, 3] = translation
            relative_est_tf_mat[:3, :3] = rotation
            cumulative_est_tf_mat = np.dot(cumulative_est_tf_mat, relative_est_tf_mat)

            # store ground truth and estimated relative poses in tum format for evaluation
            gt_cumulative_translation = cur_tf_mat[:3, 3]
            gt_cumulative_rotation = tf_trans.quaternion_from_matrix(cur_tf_mat)
            gt_cumulative_pose_tum = [t.to_sec(), gt_cumulative_translation[0], gt_cumulative_translation[1], gt_cumulative_translation[2],
                                      gt_cumulative_rotation[0], gt_cumulative_rotation[1], gt_cumulative_rotation[2], gt_cumulative_rotation[3]]
            
            est_cumulative_translation = cumulative_est_tf_mat[:3, 3]
            est_cumulative_rotation = tf_trans.quaternion_from_matrix(cumulative_est_tf_mat)
            est_cumulative_pose_tum = [t.to_sec(), est_cumulative_translation[0], est_cumulative_translation[1], est_cumulative_translation[2],
                                      est_cumulative_rotation[0], est_cumulative_rotation[1], est_cumulative_rotation[2], est_cumulative_rotation[3]]
            
            gt_poses_tum.append(gt_cumulative_pose_tum)
            est_poses_tum.append(est_cumulative_pose_tum)

            # print(f'est_rotation = {rotation}')
            # print(f'gt_rotation = {relative_gt_tf_mat[:3, :3]} \n')
            # print(f'Adjusted est_translation = {translation}')
            # print(f'gt_translation = {relative_gt_tf_mat[:3, 3]} \n')

            # print(f'relative_est_tf_mat = {relative_est_tf_mat}')
            # print(f'relative_gt_tf_mat = {relative_gt_tf_mat} \n')
            # print(f'cumulative_est_tf_mat = {cumulative_est_tf_mat}')
            # print(f'cumulative_gt_tf_mat = {cumulative_gt_tf_mat} \n')

            # add the ground truth pose to the ground truth path
            gt_pose_stamped = PoseStamped()
            gt_pose_stamped.header.stamp = rospy.Time.from_sec(t.to_sec())
            gt_pose_stamped.header.frame_id = 'map'
            gt_pose_stamped.pose.position.x = cumulative_gt_tf_mat[0, 3]
            gt_pose_stamped.pose.position.y = cumulative_gt_tf_mat[1, 3]
            gt_pose_stamped.pose.position.z = cumulative_gt_tf_mat[2, 3]
            gt_quat_array = tf_trans.quaternion_from_matrix(cumulative_gt_tf_mat)
            gt_pose_stamped.pose.orientation.x = gt_quat_array[0]
            gt_pose_stamped.pose.orientation.y = gt_quat_array[1]
            gt_pose_stamped.pose.orientation.z = gt_quat_array[2]
            gt_pose_stamped.pose.orientation.w = gt_quat_array[3]
            gt_path.poses.append(gt_pose_stamped)

            # publish the ground truth path message
            gt_path_pub.publish(gt_path)

            # add the estimated pose to the estimated path
            est_pose_stamped = PoseStamped()
            est_pose_stamped.header.stamp = rospy.Time.from_sec(t.to_sec())
            est_pose_stamped.header.frame_id = 'map'
            est_pose_stamped.pose.position.x = cumulative_est_tf_mat[0, 3]
            est_pose_stamped.pose.position.y = cumulative_est_tf_mat[1, 3]
            est_pose_stamped.pose.position.z = cumulative_est_tf_mat[2, 3]
            est_quat_array = tf_trans.quaternion_from_matrix(cumulative_est_tf_mat)
            est_pose_stamped.pose.orientation.x = est_quat_array[0]
            est_pose_stamped.pose.orientation.y = est_quat_array[1]
            est_pose_stamped.pose.orientation.z = est_quat_array[2]
            est_pose_stamped.pose.orientation.w = est_quat_array[3]
            est_path.poses.append(est_pose_stamped)

            # publish the estimated path message
            est_path_pub.publish(est_path)

        # update the previous image, timestamp and pose
        prev_img = cur_img
        prev_tf_mat = cur_tf_mat
        prev_translation = cur_translation

results_dir = 'Datasets/BotanicGarden/1005_07/pose_estimation_results/'

# delete existing results folder
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)

# create new results folder
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# define the file paths
gt_cumulative_poses_file = os.path.join(results_dir, 'Ground_Truth.txt')
est_cumulative_poses_file = os.path.join(results_dir, 'SuperPoint_SuperGlue.txt')

# save the data to the files
np.savetxt(gt_cumulative_poses_file, gt_poses_tum, delimiter=' ', fmt='%f')
np.savetxt(est_cumulative_poses_file, est_poses_tum, delimiter=' ', fmt='%f')

# evo_rpe tum Ground_Truth.txt SuperPoint_SuperGlue.txt --pose_relation point_distance_error_ratio --verbose --plot --plot_mode xz -as --delta 100 --delta_unit m
