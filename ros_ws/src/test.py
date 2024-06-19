import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
from SuperGluePretrainedNetwork.models.matching import Matching
from train import PoseEstimationModel

def test():
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

    # load the trained weights
    model_path = os.path.join('training_results/pose_estimation_model_50_epochs.pth')
    model = PoseEstimationModel().to(device)
    model.load_state_dict(torch.load(model_path))

    # move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    criterion = nn.MSELoss()
        
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
        
    bag = rosbag.Bag('Datasets/BotanicGarden/1018_13_img10hz600p.bag')

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

    # initialise cumulative transformation matrices
    cumulative_gt_tf_mat = np.eye(4)
    cumulative_est_tf_mat = np.eye(4)

    # loop through image data in rosbag
    for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw'])):
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
        cur_tf_mat[0:3, 3] = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]

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

            # convert to tensors and move to device
            mkpts0_tensor = torch.from_numpy(mkpts0).float().to(device)
            features = torch.from_numpy(mkpts1).float().unsqueeze(0).to(device)  # Add batch dimension
            mconf_tensor = torch.from_numpy(mconf).unsqueeze(-1).float().unsqueeze(0).to(device)  # Add batch dimension
            features = torch.cat((features, mconf_tensor), dim=-1)

            # create a mask with the correct shape
            features_mask = torch.ones(features.size(0), features.size(1), dtype=torch.bool).to(device)

            # calculate the ground truth relative translation and rotation
            relative_gt_tf_mat = np.dot(np.linalg.inv(prev_tf_mat), cur_tf_mat)
            cumulative_gt_tf_mat = np.dot(cumulative_gt_tf_mat, relative_gt_tf_mat)
            translation_gt = relative_gt_tf_mat[0:3, 3]
            rotation_gt = tf_trans.quaternion_from_matrix(relative_gt_tf_mat)

            # convert np arrays to tensors and move to device
            translation_gt = torch.from_numpy(translation_gt).float().to(device)
            rotation_gt = torch.from_numpy(rotation_gt).float().to(device)

            # forward pass
            translation_pred, rotation_pred = model(features, features_mask)

            # compute the loss
            loss_translation = criterion(translation_pred, translation_gt)
            loss_rotation = criterion(rotation_pred, rotation_gt)
            loss = loss_translation + loss_rotation

            translation_pred = translation_pred[0].cpu().numpy()
            translation_pred[2] = 0
            rotation_pred = rotation_pred[0].cpu().numpy()
            rotation_pred[2] = 0

            # calculate the estimated relative translation and rotation
            relative_est_tf_mat = tf_trans.quaternion_matrix(rotation_pred)
            relative_est_tf_mat[0:3, 3] = translation_pred
            cumulative_est_tf_mat = np.dot(cumulative_est_tf_mat, relative_est_tf_mat)

            print(f' loss = {loss.item()} \n')

            print(f'translation_pred = {translation_pred}')
            print(f'translation_gt = {translation_gt}')
            print(f'rotation_pred = {rotation_pred}')
            print(f'rotation_gt = {rotation_gt} \n')

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
            # print(f'gt_pose_stamped = {gt_pose_stamped} \n')

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
            # print(f'est_pose_stamped = {est_pose_stamped}')

        # update the previous image, timestamp and pose
        prev_img = cur_img
        prev_tf_mat = cur_tf_mat

if __name__ == '__main__':
    test()