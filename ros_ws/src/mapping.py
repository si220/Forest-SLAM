import rosbag
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import tf.transformations as tf_trans

bag = rosbag.Bag('Datasets/BotanicGarden/1005_00_img10hz600p.bag')

# create a visualiser
vis = o3d.visualization.Visualizer()
vis.create_window()

# initialise an empty point cloud
pcd = o3d.geometry.PointCloud()

# create a dictionary to store the ground truth poses
gt_poses = {}

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
    if index % 10 == 0:
        point_cloud = pc2.read_points(msg, skip_nans=True, field_names=('x', 'y', 'z'))
        points = np.array(list(point_cloud))

        # get the corresponding ground truth pose
        pose = find_closest_timestamp(t.to_sec(), gt_poses)
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans = tf_trans.quaternion_matrix(quat)
        trans[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        # transform the points using the ground truth pose
        points = np.dot(trans, np.vstack((points.T, np.ones(points.shape[0]))))[:3, :].T

        # create a point cloud object
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)

        # downsample the point cloud
        pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.99)

        # if pcd is empty, this is the first point cloud
        if not pcd.has_points():
            pcd.points = pcd_temp.points
            vis.add_geometry(pcd)
            print(f'map initialised')
        else:
            # add the transformed points to the overall map
            pcd.points = o3d.utility.Vector3dVector(
                np.concatenate((np.asarray(pcd.points), np.asarray(pcd_temp.points)))
            )

        # update the visualiser
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        print(f'map updated')
    
print(f'map completed')

vis.run()
