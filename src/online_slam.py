"""
Initial attempt to perform online SLAM with RealSense camera using SuperGlue feature correspondences

NOT CURRENTLY WORKING
-> INCORRECT POSE ESTIMATIONS
-> INCORRECT MAPPING

SLAM.match_frames() contains code from SuperGluePretrainedNetwork/match_pairs.py
to match SuperPoint features in pairs of rgb frames using SuperGlue
Magic Leap, Inc. remains the owner of that code
See SuperGluePretrainedNetwork for original authors implementation

SLAM.get_camera_intrinsics() gets the camera matrix from the intrinsic params

inputs:
    input_dir (path to input folder containing rgb frames) -> string
    output_dir (path to output folder to store feature correspondences) -> string
    cam_intrinsic_params (path to json file containing camera intrinsic parameters) -> string

    resize (dimensions to resize img to before feature matching) -> list of ints
    resize_float (flag that tells read_image if uint8 should be cast to float before resizing) -> bool
    rotation (the angle at which the input img has been rotated) -> int
    nms_radius (SuperPoint Non Maximum Suppression (NMS) radius) -> positive int
    keypoint_threshold (SuperPoint keypoint detector confidence threshold) -> float
    max_keypoints (max number of keypoints detected by SuperPoint, -1 keeps all keypoints) -> int
    superglue (SuperGlue weights, either 'indoor' or 'outdoor') -> string
    sinkhorn_iterations (num of Sinkhorn iterations performed by SuperGlue) -> int
    match_threshold (SuperGlue match threshold) -> float

outputs:
    point cloud file containing final trajectory (saved in output_dir)

Copyright (C) 2024  Saifullah Ijaz

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Saifullah Ijaz
date: 05/03/2024
"""

from globals import *

sys.path.append('..')

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

matplotlib.use('TkAgg')

# inputs
img_width = 640
img_height = 480
fps = 30
cam_intrinsic_params = 'intrinsics.json'

# see SuperGluePretrainedNetwork/match_pairs.py to understand hyperparams
resize = [640, 480]
resize_float = False
rotation = 0
nms_radius = 4
keypoint_threshold = 0.005
max_keypoints = 1024
superglue = 'outdoor'
sinkhorn_iterations = 20
match_threshold = 0.2

class SLAM:
    def __init__(self, img_width, img_height, fps, cam_intrinsic_params, resize, resize_float,
                 rotation, nms_radius, keypoint_threshold, max_keypoints,superglue,
                 sinkhorn_iterations, match_threshold):
        
        # params for intel realsense camera
        self.img_width = img_width
        self.img_height = img_height
        self.fps = fps

        # get camera intrinsic parameters from json file
        self.intrinsics = self.get_camera_intrinsics(cam_intrinsic_params)

        # initialise visualiser
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # hyperparams
        self.resize = resize
        self.resize_float = resize_float
        self.rotation = rotation
        self.nms_radius = nms_radius
        self.keypoint_threshold = keypoint_threshold
        self.max_keypoints = max_keypoints
        self.superglue = superglue
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold

        # initialise variable to store previous frame
        self.prev_frame = None
        self.first_frame_received = False

        # Initialise previous translation vector
        self.prev_translation = None

        # Initialise the point cloud map
        self.point_cloud_map = o3d.geometry.PointCloud()

    def get_camera_intrinsics(self, intrinsics_file):
        # load the intrinsic parameters from the json file
        with open(intrinsics_file, 'r') as f:
            intrinsics_dict = json.load(f)

        # extract intrinsic parameters
        fx = intrinsics_dict["fx"]
        fy = intrinsics_dict["fy"]
        ppx = intrinsics_dict["ppx"]
        ppy = intrinsics_dict["ppy"]

        # get the camera matrix
        camera_matrix = np.array([[fx, 0, ppx],
                                [0, fy, ppy],
                                [0, 0, 1]])

        return camera_matrix
    
    def get_frame(self, pipeline):
        # get frames from RealSense camera
        frames = pipeline.wait_for_frames()

        # get RGB and depth frames
        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # convert frames to numpy arrays
        rgb_array = np.asanyarray(rgb_frame.get_data())
        depth_array = np.asanyarray(depth_frame.get_data())

        # check if IMU data is available
        imu_frame = frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)

        if imu_frame:
            imu_data = imu_frame.as_motion_frame().get_motion_data()
            return rgb_array, depth_array, imu_data
        
        else:
            print("IMU data frame is null.")
            return rgb_array, depth_array, None

    def match_frames(self, frame_0, frame_1, device):
        # disable gradient calculations for faster inference
        torch.set_grad_enabled(False)

        # SuperPoint and SuperGlue hyperparams
        model_config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }

        # initialise SuperPoint and SuperGlue
        network = Matching(model_config).eval().to(device)

        # perform forward pass to get predictions
        model_pred = network({'image0': frame_0, 'image1': frame_1})
        model_pred = {j: k[0].cpu().numpy() for j, k in model_pred.items()}

        # get SuperPoint keypoints
        keypoints_0, keypoints_1 = model_pred['keypoints0'], model_pred['keypoints1']

        # get number of matches and match confidence from SuperGlue
        num_matches, match_confidence = model_pred['matches0'], model_pred['matching_scores0']

        # only keep features that have been matched
        valid = num_matches > -1
        matched_keypoints_0 = keypoints_0[valid]
        matched_keypoints_1 = keypoints_1[num_matches[valid]]

        # store outputs in dictionary
        feature_correspondences = {
            'keypoints0': matched_keypoints_0,
            'keypoints1': matched_keypoints_1,
            'num_matches': num_matches[valid],
            'match_confidence': match_confidence[valid]
        }

        return feature_correspondences

    def perform_slam(self):
        # use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'device = {device}')

        # initialise list to store all camera poses
        poses = []

        # initialise RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.img_width, self.img_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.img_width, self.img_height, rs.format.rgb8, self.fps)
        pipeline.start(config)

        while True:
            # get RGB and depth frames from RealSense camera
            rgb_img, depth_img, imu_data = self.get_frame(pipeline)

            # convert frames to greyscale
            grey_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

            # skip processing for the first frame
            if not self.first_frame_received:
                self.first_frame_received = True
                self.prev_frame = grey_img.copy()
                continue

            # perform SuperGlue feature correspondence
            correspondences = self.match_frames(
                torch.from_numpy(self.prev_frame/255.).float()[None, None].to(device),
                torch.from_numpy(grey_img/255.).float()[None, None].to(device),
                device
            )

            # update prev_frame to current frame
            self.prev_frame = grey_img.copy()

            # get feature correspondences
            kp0 = correspondences['keypoints0']
            kp1 = correspondences['keypoints1']

            # convert keypoints to homogeneous coordinates
            pts0 = np.column_stack((kp0, np.ones(len(kp0))))
            pts1 = np.column_stack((kp1, np.ones(len(kp1))))

            # use depth data to estimate 3D-2D correspondences
            pts1_3d = []
            valid_pts0 = []
            for i, pt in enumerate(pts1):
                u, v, _ = pt
                depth = depth_img[int(v), int(u)]
                pts1_3d.append([u, v, depth])
                valid_pts0.append(pts0[i])
            pts1_3d = np.array(pts1_3d)
            valid_pts0 = np.array(valid_pts0)

            # use solvePnPRansac to estimate camera pose
            if len(pts1_3d) >= 4:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(pts1_3d, valid_pts0[:, :2], self.intrinsics, None,
                                                            confidence=0.99, reprojectionError=8.0, 
                                                            flags=cv2.SOLVEPNP_EPNP)
            else:
                print("Not enough valid correspondences for pose estimation.")
                continue

            # update camera pose list
            pose = np.eye(4)
            pose[:3, :3] = cv2.Rodrigues(rvec)[0]
            pose[:3, 3] = tvec.flatten()
            self.pose = pose
            poses.append(pose)

            # update the trajectory based on the accumulated camera poses
            trajectory = np.array([pose[:3, 3] for pose in poses])

            # update the 3D map using depth data and camera pose
            self.update_point_cloud_map(rgb_img, depth_img, rvec, tvec, kp0)

            # visualisation
            self.visualise(trajectory, self.point_cloud_map)
            
    def update_point_cloud_map(self, rgb_img, depth_img, rvec, tvec, keypoints):
        # create a pyrealsense2.intrinsics object
        intrinsics = rs.intrinsics()
        intrinsics.width = self.img_width
        intrinsics.height = self.img_height
        intrinsics.ppx = self.intrinsics[0, 2]
        intrinsics.ppy = self.intrinsics[1, 2]
        intrinsics.fx = self.intrinsics[0, 0]
        intrinsics.fy = self.intrinsics[1, 1]
        intrinsics.model = rs.distortion.none

        # create a new point cloud for this frame
        point_cloud = o3d.geometry.PointCloud()

        # compute rotation matrix from rotation vector
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        for kp in keypoints:
            # get pixel coordinates from keypoints
            u, v = kp

            # convert depth image to point cloud
            depth = depth_img[int(v), int(u)]

            points = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth)

            # transform points to camera coordinate system
            points_camera = np.array([points[0] - tvec[0], points[1] - tvec[1], points[2] - tvec[2]])

            # rotate points to align with camera orientation
            points_camera_rotated = np.dot(rotation_matrix, points_camera)

            # transform points to world coordinate system based on the current pose
            points_world = np.dot(self.pose[:3, :3], points_camera_rotated) + self.pose[:3, 3]

            for point in points_world:
                # add the transformed points to the point cloud
                point_cloud.points.append(point)

                # get color from the RGB image
                colour = rgb_img[int(v), int(u)]
                
                # ensure colour has the correct shape
                if len(colour) == 3:
                    # normalise to [0, 1] range
                    colour = colour.astype(np.float32) / 255.0

                point_cloud.colors.append(colour)

        # update the point cloud map by adding the new point cloud
        self.point_cloud_map += point_cloud

    def visualise(self, trajectory, point_cloud_map):
        if len(trajectory > 0):
            # add trajectory line
            trajectory_line = o3d.geometry.LineSet()
            trajectory_line.points = o3d.utility.Vector3dVector(trajectory)
            trajectory_line.lines = o3d.utility.Vector2iVector(np.column_stack((np.arange(len(trajectory) - 1), np.arange(1, len(trajectory)))))
            green_colour = [0, 1, 0]
            line_colours = np.tile(np.array(green_colour), (len(trajectory) - 1, 1)).astype(np.float64)
            trajectory_line.colors = o3d.utility.Vector3dVector(line_colours)
            self.vis.add_geometry(trajectory_line)

            # add point cloud map
            self.vis.add_geometry(point_cloud_map)

            # Update visualiser
            self.vis.update_geometry(trajectory_line)
            self.vis.update_geometry(point_cloud_map)
            self.vis.poll_events()
            self.vis.update_renderer()
    
if __name__ == "__main__":
    # instantiate SLAM object
    loc = SLAM(img_width, img_height, fps, cam_intrinsic_params, resize, resize_float,
                       rotation, nms_radius, keypoint_threshold, max_keypoints,superglue,
                       sinkhorn_iterations, match_threshold)

    # perform SLAM
    loc.perform_slam()