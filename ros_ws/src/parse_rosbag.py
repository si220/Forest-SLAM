import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import message_filters
import rospy
from sensor_msgs.msg import Image, PointCloud2, Imu
from geometry_msgs.msg import PoseStamped

# camera intrinsic parameters for dalsa_rgb0_down camera from BotanicGarden
intrinsic_matrix = np.array([[642.9165664800531, 0, 460.1840658156501],
                             [0, 641.9171825800378, 308.5846449100310],
                             [0, 0, 1]])

distortion_coeffs = np.array([-0.060164620903866, 0.094005180631043, 0.0, 0.0])

# extrinsic parameters (VLP16 LIDAR to camera)
T_rgb0_vlp16 = np.array([
    [0.0238743541600432, -0.999707744440396, 0.00360642510766516, 0.138922870923538],
    [-0.00736968896588375, -0.00378431903190059, -0.999965147452649, -0.177101909101325],
    [0.999687515506770, 0.0238486947027063, -0.00745791352160211, -0.126685267545513],
    [0.0, 0.0, 0.0, 1.0]
])

bag = rosbag.Bag('Datasets/BotanicGarden/1018_00_img10hz600p.bag')
bridge = CvBridge()

# sensor topics
image_sub = message_filters.Subscriber('/dalsa_rgb/left/image_raw', Image)
lidar_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
imu_sub = message_filters.Subscriber('/imu/data', Imu)
pose_sub = message_filters.Subscriber('/gt_poses', PoseStamped)

# function to undistort and resize image
def process_image(image_msg):
    # convert ros image message to opencv image
    cv_img = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

    # resize the image
    cv_img_resized = cv2.resize(cv_img, (960, 600))

    # undistort the image
    cv_img_undistorted = cv2.undistort(cv_img_resized, intrinsic_matrix, distortion_coeffs)

    # convert the image to greyscale
    cv_img_grey = cv2.cvtColor(cv_img_undistorted, cv2.COLOR_BGR2GRAY)

    return cv_img_grey

# function to project 3D points onto 2D image plane
def project_points(points_3d, intrinsic_matrix):
    # divide by the third value to perform perspective division
    points_2d = np.dot(intrinsic_matrix, points_3d.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2, np.newaxis]

    return np.round(points_2d).astype(int)

def callback(image, lidar, imu, pose):
    # get processed image
    cv_img_grey = process_image(image)

    # get LIDAR data as points
    gen = pc2.read_points(lidar, skip_nans=True, field_names=('x', 'y', 'z'))
    points = np.array(list(gen))

    # transform LIDAR points to camera coordinate system
    # convert to homogeneous coordinates
    points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
    points_camera = np.dot(T_rgb0_vlp16, points_hom.T).T

    # project points onto image plane
    points_image = project_points(points_camera, intrinsic_matrix)

    # create a depth map (initialise with zeros, and the same size as the image)
    depth_map = np.zeros(cv_img_grey.shape, dtype=np.float32)

    # get depth data for each pixel
    for i, point in enumerate(points_image):
        x, y = point[0], point[1]
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            # z value is the depth
            depth_map[y, x] = points_camera[i, 2]

    # normalise the depth map for display
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_display = np.uint8(depth_map_normalized)

    # apply a colourmap to the depth map for visualisation
    depth_map_coloured = cv2.applyColorMap(depth_map_display, cv2.COLORMAP_JET)

    # stack both images horizontally
    images_combined = np.hstack((cv_img_grey, depth_map_coloured))

    # display the combined image
    cv2.imshow('Image and Depth Map Side by Side', images_combined)
    cv2.waitKey(0)

    # IMU
    print(imu.orientation)

    # POSE
    print(pose.pose)
    # timestamp
    print(pose.header.stamp)

    # exit after processing the first set of synchronised messages
    rospy.signal_shutdown('Done')

# ensure rostopic msgs are synced within 0.1s (use a buffer of 30 msgs)
ts = message_filters.ApproximateTimeSynchronizer([image_sub, lidar_sub, imu_sub, pose_sub], 30, 0.1, allow_headerless=True)
ts.registerCallback(callback)

rospy.init_node('BotanicGarden')
rospy.spin()
