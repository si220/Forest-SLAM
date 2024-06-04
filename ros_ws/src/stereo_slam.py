import torch
import rosbag
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import tf.transformations as tf_trans
import cv2
import numpy as np
import os
from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# disable gradient calculations as not needed for inference
torch.set_grad_enabled(False)

bridge = CvBridge()

# initialise ros publisher
rospy.init_node('path_publisher')
map_pub = rospy.Publisher('slam_map', PointCloud2, queue_size=10)
est_path_pub = rospy.Publisher('est_trajectory', Path, queue_size=10)

# initialise an empty estimated path message
est_path = Path()
est_path.header.frame_id = 'map'

bag = rosbag.Bag('Datasets/BotanicGarden/1018_13_img10hz600p.bag')

# camera intrinsic params for left image
K0 = np.array([[642.9165664800531, 0., 460.1840658156501],
                    [0., 641.9171825800378, 308.5846449100310],
                    [0., 0., 1.]])

# distortion params for left image
k1_l = -0.060164620903866
k2_l = 0.094005180631043
p1_l = 0.0
p2_l = 0.0

dist_coeffs_l = np.array([k1_l, k2_l, p1_l, p2_l, 0])

# camera intrinsic params for the right camera
K1 = np.array([[644.4385505412966, 0., 455.1775919513420],
               [0., 643.5879520187435, 304.1616226347153],
               [0., 0., 1.]])

# distortion params for the right camera
k1_r = -0.057705696896734
k2_r = 0.086955444511364
p1_r = 0.0
p2_r = 0.0

dist_coeffs_r = np.array([k1_r, k2_r, p1_r, p2_r, 0])


# RGB1 in RGB0 coordinates
T_rgb0_rgb1 = np.array([[0.999994564612669,-0.00327143011166783,-0.000410475508767800,0.253736175410149,
          0.00326819763481066,0.999965451959397,-0.00764289028177120,-0.000362553856124796,
          0.000435464509051199,0.00764150722461529,0.999970708440001,-0.000621002717451192,
          0.0,0.0,0.0,1.0]])

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

# initialise previous image
prev_img_left = None
prev_img_right = None
prev_img_left_tensor = None
prev_img_right_tensor = None

# initialise trajectory
cumulative_est_tf_mat = np.eye(4)

# initialise lists to store estimated poses in TUM format
est_poses_tum = []

frame_interval = 1

def get_disparity_map(left_image, right_image):
    matcher = cv2.StereoSGBM_create(numDisparities=6 * 16,
                                    minDisparity=0,
                                    blockSize=7,
                                    P1=8 * 7 ** 2,
                                    P2=32 * 7 ** 2,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                    )

    disparity_map = matcher.compute(left_image, right_image).astype(np.float32) / 16

    # avoid division by zero and instability
    disparity_map[disparity_map == 0.0] = 0.1
    disparity_map[disparity_map == -1.0] = 0.1

    return disparity_map

# loop through image data in rosbag
for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw', '/dalsa_rgb/right/image_raw'])):
    print(f'index: {index}')

    # exit on ctrl + c
    if rospy.is_shutdown():
        break

    if topic == '/dalsa_rgb/left/image_raw':
        cur_img_left = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cur_img_left = cv2.undistort(cur_img_left, K0, dist_coeffs_l)
        cur_img_left = cv2.cvtColor(cur_img_left, cv2.COLOR_BGR2GRAY)
        cur_img_left_tensor = torch.from_numpy(cur_img_left/255.).float()[None, None].to(device)

    elif topic == '/dalsa_rgb/right/image_raw':
        cur_img_right = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cur_img_right = cv2.undistort(cur_img_right, K1, dist_coeffs_r)
        cur_img_right = cv2.cvtColor(cur_img_right, cv2.COLOR_BGR2GRAY)
        cur_img_right_tensor = torch.from_numpy(cur_img_right/255.).float()[None, None].to(device)

        # select pairs that are n frames apart to ensure there is sufficient motion between frames
        if index % frame_interval == 0:
            if prev_img_left is not None and prev_img_right is not None:
                # perform the matching on the left image
                pred_left = feature_matcher({'image0': prev_img_left_tensor, 'image1': cur_img_left_tensor})
                pred_left = {k: v[0].detach().cpu().numpy() for k, v in pred_left.items()}
                kpts0_l, kpts1_l = pred_left['keypoints0'], pred_left['keypoints1']
                matches_l, conf_l = pred_left['matches0'], pred_left['matching_scores0']

                # only keep the matched keypoints
                valid_l = matches_l > -1
                mkpts0_l = kpts0_l[valid_l]
                mkpts1_l = kpts1_l[matches_l[valid_l]]

                # perform the matching on the right image
                pred_right = feature_matcher({'image0': prev_img_right_tensor, 'image1': cur_img_right_tensor})
                pred_right = {k: v[0].detach().cpu().numpy() for k, v in pred_right.items()}
                kpts0_r, kpts1_r = pred_right['keypoints0'], pred_right['keypoints1']
                matches_r, conf_r = pred_right['matches0'], pred_right['matching_scores0']

                # only keep the matched keypoints
                valid_r = matches_r > -1
                mkpts0_r = kpts0_r[valid_r]
                mkpts1_r = kpts1_r[matches_r[valid_r]]

                disparity_map = get_disparity_map(prev_img_left, prev_img_right)

                # calculate depth (Z)
                depth = np.ones(disparity_map.shape)
                cx = K0[0,2]
                cy = K0[1,2]
                fx = K0[0,0]
                fy = K0[1,1]
                baseline = np.linalg.norm(T_rgb0_rgb1[:3, 3])
                depth = fx * baseline / disparity_map

                # calculate X and Y coordinates
                X = mkpts0_l[:, 0]
                Y = mkpts0_l[:, 1]

                # get depth for each matched keypoint
                Z = depth[Y.astype(int), X.astype(int)]

                X = ((X - cx) / fx) * Z
                Y = ((Y - cy) / fy) * Z

                # create 3D points
                points3D = np.column_stack((X, Y, Z))

                # remove points with extreme depth values
                valid_depth = (Z > 0.1) & (Z < 1000)
                points3D = points3D[valid_depth]
                mkpts1_l = mkpts1_l[valid_depth]

                # check if there are at least 6 valid keypoints
                if len(points3D) >= 6:
                    # estimate pose using PnP
                    _, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(points3D, mkpts1_l, K0, dist_coeffs_l, reprojectionError=1.0,
                                                                                confidence=0.99, iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE)

                    # convert rotation vector to rotation matrix
                    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

                    # create transformation matrix
                    transformation_mat = np.eye(4)
                    transformation_mat[:3, :3] = rotation_mat
                    transformation_mat[:3, 3] = translation_vec.T

                    # update cumulative transformation matrix
                    cumulative_est_tf_mat = np.dot(cumulative_est_tf_mat, transformation_mat)

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
                    print(f'path updated at index: {index}')

                    est_poses_tum.append([t.to_sec(), cumulative_est_tf_mat[0, 3], cumulative_est_tf_mat[1, 3], cumulative_est_tf_mat[2, 3],
                                        est_quat_array[0], est_quat_array[1], est_quat_array[2], est_quat_array[3]])

            prev_img_left = cur_img_left
            prev_img_right = cur_img_right
            prev_img_left_tensor = cur_img_left_tensor
            prev_img_right_tensor = cur_img_right_tensor

results_dir = 'Datasets/BotanicGarden/1018_13/pose_estimation_results/'

# create new results folder
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# save the data to the files
est_cumulative_poses_file = os.path.join(results_dir, 'SuperPoint_SuperGlue_Stereo.txt')
np.savetxt(est_cumulative_poses_file, est_poses_tum, delimiter=' ', fmt='%f')
