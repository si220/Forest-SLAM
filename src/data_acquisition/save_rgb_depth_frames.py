"""
script to save RGB and depth data from intel realsense d455 camera into a specific folder

inputs:
    output_dir (path to folder to save data) -> string
    img_width (width of the image to be recorded by realsense camera) -> int
    img_height (height of the image to be recorded by realsense camera) -> int
    fps (frames per second to be recorded by realsense camera) -> int

outputs:
    timestamped sub-folders within output_dir containing RGB and depth frames

usage:
    connect realsense camera via USB
    modify inputs
    run script using 'python save_rgb_depth_frames.py'
    press 'r' on the keyboard to start recording
    press 's' on the keyboard to stop recording

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
date: 23/02/2024
"""

from imports import *

# inputs
output_dir = 'C:/Users/saifu/OneDrive - Imperial College London/Year 4/FYP/Forest_Data/'
img_width=1280
img_height=720
fps=30

class data_acquirer:
    def __init__(self, output_dir, img_width=1280, img_height=720, fps=30):
        # initialise output folder location, image size and frames per second
        self.output_dir = output_dir
        self.img_width = img_width
        self.img_height = img_height
        self.fps = fps

    # function to create folder
    def create_folder(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    def record_data(self):
        # set initial frame number to 0
        frame_number = 0

        # create timestamped folder to save data
        output_folder = os.path.join(self.output_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.create_folder(output_folder)

        # create subfolders for rgb and depth frames
        rgb_folder = os.path.join(output_folder, 'rgb')
        depth_folder = os.path.join(output_folder, 'depth')
        self.create_folder(rgb_folder)
        self.create_folder(depth_folder)

        # configure intel realsense d455 camera
        pipeline = rs.pipeline()
        config = rs.config()

        # enable rgb stream
        config.enable_stream(rs.stream.color, self.img_width, self.img_height, rs.format.bgr8, self.fps)

        # enable depth stream
        config.enable_stream(rs.stream.depth, self.img_width, self.img_height, rs.format.z16, self.fps)

        # start camera stream
        pipeline.start(config)

        try:
            while True:
                # wait for a pair of rgb and depth frames
                frames = pipeline.wait_for_frames()

                # get rgb and depth frames
                rgb_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                # if either frame do not arrive move to next iteration
                if not rgb_frame or not depth_frame:
                    continue

                # convert images to numpy arrays
                rgb_image = np.asanyarray(rgb_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # save frames
                cv2.imwrite(os.path.join(rgb_folder, f"{frame_number:05d}.png"), rgb_image)
                cv2.imwrite(os.path.join(depth_folder, f"{frame_number:05d}.png"), depth_image)

                frame_number += 1

        # exits if user presses 'ctrl + c'
        except KeyboardInterrupt:
            pass

        finally:
            # stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()

rs_camera = data_acquirer(output_dir, img_width, img_height, fps)

rs_camera.record_data()
