import torch
import rosbag
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import tf.transformations as tf_trans
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# disable gradient calculations as not needed for inference
torch.set_grad_enabled(False)

bridge = CvBridge()

# initialise ros publisher
rospy.init_node('path_publisher')
gt_path_pub = rospy.Publisher('gt_trajectory', Path, queue_size=10)

# initialise an empty ground truth path message
gt_path = Path()
gt_path.header.frame_id = 'map'

bag = rosbag.Bag('Datasets/BotanicGarden/1018_13_img10hz600p.bag')

# extrinsic params for the transformation from RGB0 to VLP16 coordinate frame
T_rgb0_vlp16 = np.array([[0.0238743541600432, -0.999707744440396, 0.00360642510766516, 0.138922870923538],
                          [-0.00736968896588375, -0.00378431903190059, -0.999965147452649, -0.177101909101325],
                          [0.999687515506770, 0.0238486947027063, -0.00745791352160211, -0.126685267545513],
                          [0.0, 0.0, 0.0, 1.0]])

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

# initialise previous pose
prev_tf_mat = None

# initialise trajectory
cumulative_gt_tf_mat = np.eye(4)

# initialise lists to store ground truth and estimated poses in TUM format
gt_poses_tum = []

# loop through image data in rosbag
for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw'])):
    # exit on ctrl + c
    if rospy.is_shutdown():
        break

    # get the corresponding ground truth pose
    cur_pose = find_closest_timestamp(t.to_sec(), gt_poses)
    cur_quat = [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]
    cur_translation = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]
    cur_tf_mat = tf_trans.quaternion_matrix(cur_quat)
    cur_tf_mat[0:3, 3] = cur_translation
    cur_tf_mat = np.dot(T_rgb0_vlp16, cur_tf_mat)

    if prev_tf_mat is not None:
        # get the ground truth relative and cumulative poses
        relative_gt_tf_mat = np.dot(np.linalg.inv(prev_tf_mat), cur_tf_mat)
        cumulative_gt_tf_mat = np.dot(cumulative_gt_tf_mat, relative_gt_tf_mat)

        # store ground truth and estimated relative poses in tum format for evaluation
        gt_cumulative_translation = cur_tf_mat[:3, 3]
        gt_cumulative_rotation = tf_trans.quaternion_from_matrix(cur_tf_mat)
        gt_cumulative_pose_tum = [t.to_sec(), gt_cumulative_translation[0], gt_cumulative_translation[1], gt_cumulative_translation[2],
                                    gt_cumulative_rotation[0], gt_cumulative_rotation[1], gt_cumulative_rotation[2], gt_cumulative_rotation[3]]
                
        gt_poses_tum.append(gt_cumulative_pose_tum)

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

    # update the previous pose
    prev_tf_mat = cur_tf_mat
    prev_translation = cur_translation

results_dir = 'Datasets/BotanicGarden/1018_13/pose_estimation_results/'

# create new results folder
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# save the data to the files
gt_cumulative_poses_file = os.path.join(results_dir, '1018_13_Ground_Truth.txt')
np.savetxt(gt_cumulative_poses_file, gt_poses_tum, delimiter=' ', fmt='%f')
