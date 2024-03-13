"""
Perform pose estimations using SuperGlue feature correspondences

Localisation.match_features() contains code from SuperGluePretrainedNetwork/match_pairs.py
to match SuperPoint features in pairs of rgb frames using SuperGlue
See SuperGluePretrainedNetwork for original authors implementation

Localisation.get_camera_intrinsics() loads the camera intrinsic parameters from a yaml file

Localisation.pose_estimation() loops through input_dir, getting feature correspondences
essential matrix and relative pose are then estimated
trajectories are visualised in real time using open3d
final trajectory is saved to output_dir

inputs:
    input_dir (path to input folder containing rgb frames) -> string
    output_dir (path to output folder to store feature correspondences) -> string
    cam_intrinsic_params (path to yaml file containing camera intrinsic parameters) -> string

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

# inputs
input_dir = 'C:/Users/saifu/source/repos/Forest-SLAM/forest_data/1018_dalsa_garden_short/1018_garden_short_imgs/00/c54d7a_png'
output_dir = 'C:/Users/saifu/source/repos/Forest-SLAM/forest_data/1018_dalsa_garden_short/1018_garden_short_imgs/00/c54d7a_feature_correspondences'
cam_intrinsic_params = 'C:/Users/saifu/source/repos/Forest-SLAM/src/dalsa_rgb0.yaml'

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

class Localisation:
    def __init__(self, input_dir, output_dir, cam_intrinsic_params, resize, resize_float,
                 rotation, nms_radius, keypoint_threshold, max_keypoints,superglue,
                 sinkhorn_iterations, match_threshold):
        
        self.input_dir = input_dir
        self.output_dir = output_dir

        # get camera intrinsic parameters from yaml file
        self.intrinsics = self.get_camera_intrinsics(cam_intrinsic_params)

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

    def get_camera_intrinsics(self, intrinsics_file):
        with open(intrinsics_file, 'r') as file:
            intrinsics = yaml.safe_load(file)

        return intrinsics

    def match_frames(self, frame1_path, frame2_path, device):
        # disable gradient calculations for faster inference
        torch.set_grad_enabled(False)

        # load the img pair using read_image function from SuperGluePretrainedNetwork.models.utils
        image0, inp0, scales0 = read_image(frame1_path, device, self.resize, self.rotation, self.resize_float)
        image1, inp1, scales1 = read_image(frame2_path, device, self.resize, self.rotation, self.resize_float)

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
        model_pred = network({'image0': inp0, 'image1': inp1})
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

    def pose_estimation(self):
        # use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'device = {device}')

        # initialise visualiser
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # initialise list to store all translation vectors
        translations = []

        # ensure input images are in correct order
        files = sorted(os.listdir(self.input_dir))

        # loop through input directory
        for i in range(len(files) - 1):
            # get path to adjacent frames
            frame1_path = os.path.abspath(os.path.join(self.input_dir, files[i]))
            frame2_path = os.path.abspath(os.path.join(self.input_dir, files[i + 1]))

            # perform SuperGlue feature correspondence
            correspondences = self.match_frames(frame1_path, frame2_path, device)

            # extract feature correspondences
            kp0 = correspondences['keypoints0']
            kp1 = correspondences['keypoints1']

            # convert keypoints to homogeneous coordinates
            pts0 = np.column_stack((kp0, np.ones(len(kp0))))
            pts1 = np.column_stack((kp1, np.ones(len(kp1))))

            # estimate the essential matrix
            E, _ = cv2.findEssentialMat(pts0[:, :2], pts1[:, :2], focal=self.intrinsics['projection_parameters']['fx'],
                                         pp=(self.intrinsics['projection_parameters']['cx'],
                                             self.intrinsics['projection_parameters']['cy']),
                                         method=cv2.RANSAC)

            # recover the relative pose (rotation and translation) from the essential matrix
            _, R, t, _ = cv2.recoverPose(E, pts0[:, :2], pts1[:, :2], focal=self.intrinsics['projection_parameters']['fx'],
                                          pp=(self.intrinsics['projection_parameters']['cx'],
                                              self.intrinsics['projection_parameters']['cy']))
            
            # update list with translation vector
            translations.append(t.flatten())

            # convert translation vectors to 3D coordinates
            trajectory = np.cumsum(np.array(translations), axis=0)

            # check trajectory contains more than 1 point before creating lineset
            if len(trajectory) > 1:
                # create a lineset from the trajectory
                lineset = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(trajectory),
                    lines=o3d.utility.Vector2iVector(list(zip(range(len(trajectory) - 1), range(1, len(trajectory)))))
                )

                # update the visualiser
                vis.remove_geometry(lineset)
                vis.add_geometry(lineset)
                vis.update_geometry(lineset)
                vis.poll_events()
                vis.update_renderer()

        # create a point cloud from the trajectory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(trajectory)

        # save the point cloud in output_dir
        o3d.io.write_point_cloud(os.path.join(self.output_dir, "1018-00-trajectory.pcd"), pcd)

        # start the event loop of the visualiser
        vis.run()

        # kill the visualiser window
        vis.destroy_window()
    
if __name__ == "__main__":
    # instantiate Localisation object
    loc = Localisation(input_dir, output_dir, cam_intrinsic_params, resize, resize_float,
                       rotation, nms_radius, keypoint_threshold, max_keypoints,superglue,
                       sinkhorn_iterations, match_threshold)

    # perform pose estimation
    loc.pose_estimation()