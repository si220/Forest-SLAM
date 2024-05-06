import rosbag
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import tf.transformations as tf_trans
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

rospy.init_node('slam_publisher')
map_pub = rospy.Publisher('slam_map', PointCloud2, queue_size=10)
path_pub = rospy.Publisher('trajectory', Path, queue_size=10)

bag = rosbag.Bag('Datasets/BotanicGarden/1018_13_img10hz600p.bag')

# initialise an empty point cloud
pcd = o3d.geometry.PointCloud()

# create a dictionary to store the ground truth poses
gt_poses = {}

# initialise an empty Path message
path = Path()
path.header.frame_id = 'map'

# loop through ground truth poses in rosbag
for topic, msg, t in bag.read_messages(topics=['/gt_poses']):
    gt_poses[t.to_sec()] = msg.pose

def find_closest_timestamp(target, timestamp_dict):
    # get the list of timestamps
    timestamps = np.array(list(timestamp_dict.keys()))

    # find the index of the closest timestamp
    idx = np.argmin(np.abs(timestamps - target))

    # return the pose corresponding to the closest timestamp
    return timestamp_dict[timestamps[idx]]

# loop through lidar data in rosbag
for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/velodyne_points'])):
    # exit on ctrl + c
    if rospy.is_shutdown():
        break

    if index % 10 == 0:
        point_cloud = pc2.read_points(msg, skip_nans=True, field_names=('x', 'y', 'z'))
        points = np.array(list(point_cloud))

        # get the corresponding ground truth pose
        pose = find_closest_timestamp(t.to_sec(), gt_poses)
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        transformation_matrix = tf_trans.quaternion_matrix(quaternion)
        transformation_matrix[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        # transform the points using the ground truth pose
        points = np.dot(transformation_matrix, np.vstack((points.T, np.ones(points.shape[0]))))[:3, :].T

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

        # add the pose to the path
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.from_sec(t.to_sec())
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose = pose
        path.poses.append(pose_stamped)

        # publish the path message
        path_pub.publish(path)
        print(f'path updated')
    
print(f'map completed')
