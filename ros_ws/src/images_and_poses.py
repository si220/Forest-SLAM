import cv2
import numpy as np
import os
import shutil
import rosbag
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
from cv_bridge import CvBridge

bag = rosbag.Bag('Datasets/BotanicGarden/1018_13_img10hz600p.bag')
bridge = CvBridge()

# path to store outputs
img_folder = 'Datasets/BotanicGarden/1018_13/greyscale_imgs/'
pose_txt_file = 'Datasets/BotanicGarden/1018_13/poses.txt'
SuperGlue_matches = 'Datasets/BotanicGarden/1018_13/SuperGlue_matches_tuned/'

# delete the old folder if it exists
if os.path.exists(img_folder):
    shutil.rmtree(img_folder, ignore_errors=True)

if os.path.exists(SuperGlue_matches):
    shutil.rmtree(SuperGlue_matches, ignore_errors=True)

# create a new output folder
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# delete the old file if it exists
if os.path.exists(pose_txt_file):
    os.remove(pose_txt_file)

# create a new output file
if not os.path.exists(pose_txt_file):
    with open(pose_txt_file, 'w') as f:
        pass

# camera intrinsic parameters for dalsa_rgb0_down camera from BotanicGarden
intrinsic_matrix = np.array([[642.9165664800531, 0, 460.1840658156501],
                             [0, 641.9171825800378, 308.5846449100310],
                             [0, 0, 1]])

distortion_coeffs = np.array([-0.060164620903866, 0.094005180631043, 0.0, 0.0])

# VLP16 to RGB0 transformation matrix
T_rgb0_vlp16 = np.array([[0.0238743541600432,-0.999707744440396,0.00360642510766516,0.138922870923538],
                         [-0.00736968896588375,-0.00378431903190059,-0.999965147452649 ,-0.177101909101325],
                         [0.999687515506770,0.0238486947027063,-0.00745791352160211,-0.126685267545513],
                         [0.0,0.0,0.0,1.0]])

# initialise dictionaries for images and poses
images = {}
poses = {}

# loop through ros msgs in bag file
for topic, msg, t in bag.read_messages(topics=['/dalsa_rgb/left/image_raw', '/gt_poses']):
    timestamp = t.to_sec()

    if topic == '/dalsa_rgb/left/image_raw':
        # convert ros image message to opencv image
        rgb_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # undistort the image
        rgb_img_undistorted = cv2.undistort(rgb_img, intrinsic_matrix, distortion_coeffs)

        # convert the image to greyscale
        grey_img_undistorted = cv2.cvtColor(rgb_img_undistorted, cv2.COLOR_BGR2GRAY)

        # save the image to the dictionary with timestamp as key
        images[timestamp] = grey_img_undistorted

    elif topic == '/gt_poses':
        # get the pose
        pose = msg.pose

        # convert the pose to TUM format
        t = pose.position
        q = pose.orientation
        tum_pose = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]

        # form pose matrix using translations and rotations
        B = np.eye(4)
        B[:3, 3] = tum_pose[:3]
        B[:3, :3] = tf_trans.quaternion_matrix(tum_pose[3:])[:3, :3]

        # transform the pose from the LIDAR frame to the camera frame
        T = np.linalg.inv(T_rgb0_vlp16).dot(B)

        # save the pose to the dictionary with timestamp as key
        poses[timestamp] = T

# initialise the previous pose and image path
T_prev = None
img_path_prev = None

# initialise a counter for the images
i = 0

# sort the timestamps
timestamps_images = sorted(images.keys())
timestamps_poses = np.array(sorted(poses.keys()))

# iterate over the sorted timestamps of images
for timestamp_image in timestamps_images:
    # get the image for this timestamp
    grey_img_undistorted = images[timestamp_image]

    # save the image to the output folder
    img_path = os.path.join(img_folder, f'grey_img_undistorted_{i}.png')
    cv2.imwrite(img_path, grey_img_undistorted)

    cur_path = f'grey_img_undistorted_{i}.png'

    # find the closest pose timestamp
    idx = (np.abs(timestamps_poses - timestamp_image)).argmin()
    closest_timestamp_pose = timestamps_poses[idx]

    # get the pose for the closest timestamp
    T = poses[closest_timestamp_pose]

    # compute the relative pose if there is a previous pose
    if T_prev is not None:
        T_AB = np.linalg.inv(T_prev).dot(T)

        # write the SuperGlue input line for this image pair
        with open(pose_txt_file, 'a') as f:
            f.write(f'{prev_path} {cur_path} 0 0 {" ".join(map(str, intrinsic_matrix.flatten()))} {" ".join(map(str, intrinsic_matrix.flatten()))} {" ".join(map(str, T_AB.flatten()))}\n')

    # update the previous pose and image path
    T_prev = T
    prev_path = cur_path

    # increment the image counter
    i += 1

# python3 match_pairs.py --input_pairs ~/ros_ws/src/Datasets/BotanicGarden/poses.txt --input_dir ~/ros_ws/src/Datasets/BotanicGarden/greyscale_imgs/ --output_dir ~/ros_ws/src/Datasets/BotanicGarden/SuperGlue_matches/ --eval --viz --show_keypoints --superglue outdoor
# python3 match_pairs.py --input_pairs ~/ros_ws/src/Datasets/BotanicGarden/poses.txt --input_dir ~/ros_ws/src/Datasets/BotanicGarden/greyscale_imgs/ --output_dir ~/ros_ws/src/Datasets/BotanicGarden/SuperGlue_matches_tuned/ --eval --viz --show_keypoints --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float

# python3 match_pairs.py --input_pairs ~/ros_ws/src/Datasets/BotanicGarden/1018_13/poses.txt --input_dir ~/ros_ws/src/Datasets/BotanicGarden/1018_13/greyscale_imgs/ --output_dir ~/ros_ws/src/Datasets/BotanicGarden/1018_13/SuperGlue_matches/ --eval --viz --show_keypoints --superglue outdoor
# python3 match_pairs.py --input_pairs ~/ros_ws/src/Datasets/BotanicGarden/1018_13/poses.txt --input_dir ~/ros_ws/src/Datasets/BotanicGarden/1018_13/greyscale_imgs/ --output_dir ~/ros_ws/src/Datasets/BotanicGarden/1018_13/SuperGlue_matches_tuned/ --eval --viz --show_keypoints --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float