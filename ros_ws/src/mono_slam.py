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

# initialise an empty point cloud
pcd = o3d.geometry.PointCloud()

# initialise an empty estimated path message
est_path = Path()
est_path.header.frame_id = 'map'

bag = rosbag.Bag('Datasets/BotanicGarden/1018_00_img10hz600p.bag')

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
prev_img = None

# initialise trajectory
cumulative_est_tf_mat = np.eye(4)

# initialise lists to store estimated poses in TUM format
est_poses_tum = []

# set frame interval for feature matching
frame_interval = 1

# initialise latest point cloud message
latest_point_cloud_msg = None

# loop through image data in rosbag
for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw'])):
    # exit on ctrl + c
    if rospy.is_shutdown():
        break

    ################################# LOCALISATION #################################
    cur_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cur_img = cv2.undistort(cur_img, K0, dist_coeffs)
    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    cur_img = torch.from_numpy(cur_img/255.).float()[None, None].to(device)

    # select pairs that are n frames apart to ensure there is sufficient motion between frames
    if index % frame_interval == 0:
        if prev_img is not None:
            # perform the matching
            pred = feature_matcher({'image0': prev_img, 'image1': cur_img})
            pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            
            # only keep the matched keypoints
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]

            # calculate the estimated relative translation and rotation
            E, mask = cv2.findEssentialMat(mkpts0, mkpts1, focal=K0[0,0], pp=(K0[0,2], K0[1,2]), method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, rotation, translation, _ = cv2.recoverPose(E, mkpts0, mkpts1, focal=K0[0,0], pp=(K0[0,2], K0[1,2]))
            translation = translation.reshape(3)

            relative_est_tf_mat = np.eye(4)
            relative_est_tf_mat[:3, 3] = translation
            relative_est_tf_mat[:3, :3] = rotation
            cumulative_est_tf_mat = np.dot(cumulative_est_tf_mat, relative_est_tf_mat)
            
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
            print(f'path updated')

            est_poses_tum.append([t.to_sec(), cumulative_est_tf_mat[0, 3], cumulative_est_tf_mat[1, 3], cumulative_est_tf_mat[2, 3],
                                    est_quat_array[0], est_quat_array[1], est_quat_array[2], est_quat_array[3]])

            # process the latest point cloud message if it exists
            if latest_point_cloud_msg is not None:
                ################################# MAPPING #################################
                point_cloud = pc2.read_points(latest_point_cloud_msg, skip_nans=True, field_names=('x', 'y', 'z'))
                points = np.array(list(point_cloud))

                # transform the points using the ground truth pose
                points = np.dot(cumulative_est_tf_mat, np.vstack((points.T, np.ones(points.shape[0]))))[:3, :].T

                # create a point cloud object
                pcd_temp = o3d.geometry.PointCloud()
                pcd_temp.points = o3d.utility.Vector3dVector(points)

                # downsample the point cloud
                pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.5)

                # if pcd is empty, this is the first point cloud
                if not pcd.has_points():
                    pcd.points = pcd_temp.points
                else:
                    # add the transformed points to the overall map
                    pcd.points = o3d.utility.Vector3dVector(
                        np.concatenate((np.asarray(pcd.points), np.asarray(pcd_temp.points)))
                    )

                # create a header for the point cloud message
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'map'

                # convert the open3d point cloud to a ROS PointCloud2 message
                pc2_msg = pc2.create_cloud_xyz32(header, np.asarray(pcd.points))

                # publish the point cloud message
                map_pub.publish(pc2_msg)
                print(f'map updated')

                # reset the latest point cloud message
                latest_point_cloud_msg = None

        # update the previous image
        prev_img = cur_img

    elif topic == '/velodyne_points':
        latest_point_cloud_msg = msg

# uncomment the following to save the estimated and ground truth poses
# results_dir = 'Datasets/BotanicGarden/1018_00/pose_estimation_results/'

# # create new results folder
# if not os.path.exists(results_dir):
#     os.mkdir(results_dir)

# # save the data to the files
# est_cumulative_poses_file = os.path.join(results_dir, 'SuperPoint_SuperGlue_Mono.txt')
# np.savetxt(est_cumulative_poses_file, est_poses_tum, delimiter=' ', fmt='%f')
