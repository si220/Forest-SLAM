"""
Create 3D point clouds using rgb and depth images from recorded rosbag

inputs:
    rosbag_path (path to folder containing rosbag) -> string
    intrinsic_path (path to json file containing intrinsic parameters for camera) -> string

outputs:
    video playback of 3D point clouds

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

import open3d as o3d
import json

# inputs
rosbag_path = 'C:/Users/saifu/OneDrive - Imperial College London/Year 4/FYP/burnham_beeches/20240224_112232.bag'
intrinsic_path = 'intrinsics.json'

# load the intrinsic parameters from the json file
with open(intrinsic_path, 'r') as f:
    intrinsics_dict = json.load(f)

# create a PinholeCameraIntrinsic object with the loaded intrinsic parameters
intrinsics = o3d.camera.PinholeCameraIntrinsic(
    intrinsics_dict["width"],
    intrinsics_dict["height"],
    intrinsics_dict["fx"],
    intrinsics_dict["fy"],
    intrinsics_dict["ppx"],
    intrinsics_dict["ppy"]
)

# create a bag reader
bag_reader = o3d.t.io.RSBagReader()
bag_reader.open(rosbag_path)

# initialise the visualiser window
vis = o3d.visualization.Visualizer()
vis.create_window()

# transformation to flip the point clouds 180° since they were inverted in rosbag
flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

# loop over the frames in the bag file
while not bag_reader.is_eof():
    # get the next frame
    frame = bag_reader.next_frame()

    # get separate rgb and depth images
    colour_image = o3d.geometry.Image(frame.color)
    depth_image = o3d.geometry.Image(frame.depth)

    # fuse rgb and depth images to rgb-d image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colour_image,
        depth_image,
        # keep original colours from recording
        convert_rgb_to_intensity=False
    )

    # create a point cloud from the rgb-d image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )

    # flip point cloud 180°
    pcd.transform(flip_transform)

    # clear the geometries from the visualiser
    vis.clear_geometries()

    # add the new point cloud to the visualiser
    vis.add_geometry(pcd)

    # update the visualiser
    vis.poll_events()
    vis.update_renderer()
