import cv2
import rosbag
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

bag = rosbag.Bag('Datasets/BotanicGarden/1018_00_img10hz600p.bag')
bridge = CvBridge()


# IMAGE
for topic, msg, t in bag.read_messages(topics=['/dalsa_gray/left/image_raw']):
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    cv2.imshow('Image', cv_img)
    cv2.waitKey(500)

cv2.destroyAllWindows()


# LIDAR
for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
    gen = pc2.read_points(msg, skip_nans=True, field_names=('x', 'y', 'z'))
    points = list(gen)


# IMU
for topic, msg, t in bag.read_messages(topics=['/imu/data']):
    print(msg.orientation)
    

# POSE
for topic, msg, t in bag.read_messages(topics=['/gt_poses']):
    print(msg.pose)
    # timestamp
    print(msg.header.stamp)
