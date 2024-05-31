import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import rosbag
import tf.transformations as tf_trans
import rospy
import cv2
import os
import numpy as np
import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

torch.manual_seed(0)

class BotanicGarden(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(file, allow_pickle=True).item()
        img = torch.tensor(data['img']).unsqueeze(0).float()
        pose = torch.tensor(data['pose']).float()

        return img, pose

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), x.size(1), -1)
        return x

class PoseEstimationTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(PoseEstimationTransformer, self).__init__()
        self.cnn = CNNFeatureExtractor()
        self.embedding = nn.Linear(256, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )

        self.translation = nn.Linear(d_model, 3)
        self.rotation = nn.Linear(d_model, 9)

    def forward(self, imgs):
        batch_size, channels, h, w = imgs.size()
        features = self.cnn(imgs)
        features = features.transpose(1, 2)
        features = self.embedding(features)
        features = features.transpose(0, 1)
        pose = self.transformer(features)
        pose = pose[-1]
        est_translation = self.translation(pose)
        est_rotation = self.rotation(pose)
        est_rotation = est_rotation.view(est_rotation.shape[0], 3, 3)

        return est_translation, est_rotation

def train():
    # create directory to store results
    output_dir = 'training_results'

    # create new output folder
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # paths to files to store training losses
    training_losses = os.path.join(output_dir, 'training_losses.txt')

    model = PoseEstimationTransformer().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable parameters = {num_params}')

    train_data_dir = 'Datasets/BotanicGarden/training_data/'
    train_set = BotanicGarden(train_data_dir)

    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        training_loss = []

        for img, gt_pose in train_dataloader:
            img, gt_pose = img.to(device), gt_pose.to(device)

            gt_translation = gt_pose[:, :3, 3]
            gt_rotation = gt_pose[:, :3, :3]

            est_translation, est_rotation = model(img)

            translation_loss = criterion(est_translation, gt_translation)
            rotation_loss = criterion(est_rotation, gt_rotation)
            loss = translation_loss + rotation_loss

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            training_loss.append(loss.item())

        avg_training_loss = sum(training_loss) / len(training_loss)
        print(f'epoch {epoch}: training loss = {avg_training_loss} \n')

        with open(training_losses, 'a') as f:
            f.write(str(avg_training_loss) + '\n')

        print(f'est_translation = {est_translation[0].detach().cpu().numpy()}')
        print(f'gt_translation = {gt_translation[0].detach().cpu().numpy()} \n')
        print(f'est_rotation = {est_rotation[0].detach().cpu().numpy()}')
        print(f'gt_rotation = {gt_rotation[0].detach().cpu().numpy()} \n')

    torch.save(model.state_dict(), os.path.join(output_dir, f'pose_estimation_transformer_{num_epochs}_epochs.pth'))

def test():
    # disable gradient calculations as not needed for inference
    torch.set_grad_enabled(False)

    model_path = os.path.join('training_results/pose_estimation_transformer_30_epochs.pth')

    model = PoseEstimationTransformer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    bridge = CvBridge()

    # initialise ros publisher
    rospy.init_node('path_publisher')
    gt_path_pub = rospy.Publisher('gt_trajectory', Path, queue_size=10)
    est_path_pub = rospy.Publisher('est_trajectory', Path, queue_size=10)

    # initialise an empty ground truth path message
    gt_path = Path()
    gt_path.header.frame_id = 'map'

    # initialise an empty estimated path message
    est_path = Path()
    est_path.header.frame_id = 'map'

    bag = rosbag.Bag('Datasets/BotanicGarden/1006_01_img10hz600p.bag')

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

    # extrinsic params for the transformation from RGB0 to VLP16 coordinate frame
    T_rgb0_vlp16 = np.array([[0.0238743541600432, -0.999707744440396, 0.00360642510766516, 0.138922870923538],
                            [-0.00736968896588375, -0.00378431903190059, -0.999965147452649, -0.177101909101325],
                            [0.999687515506770, 0.0238486947027063, -0.00745791352160211, -0.126685267545513],
                            [0.0, 0.0, 0.0, 1.0]])
    
    # create a dictionary to store the ground truth poses
    gt_poses = {}

    # loop through ground truth poses in rosbag
    for topic, msg, t in bag.read_messages(topics=['/gt_poses']):
        gt_poses[t.to_sec()] = msg.pose

    # function to find current pose based on closest timestamp
    def find_closest_timestamp(target, timestamp_dict):
        # get the list of timestamps
        timestamps = np.array(list(timestamp_dict.keys()))

        # find the index of the closest timestamp
        idx = np.argmin(np.abs(timestamps - target))

        # return the pose corresponding to the closest timestamp
        return timestamp_dict[timestamps[idx]]
    
    # initialise previous image, timestamp, ground truth pose and transformation matrix
    prev_img = None
    prev_tf_mat = None
    prev_translation = None

    # initialise transformation matrices
    cumulative_gt_tf_mat = np.eye(4)
    cumulative_est_tf_mat = np.eye(4)

    # loop through image data in rosbag
    for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw'])):
        # exit on ctrl + c
        if rospy.is_shutdown():
            break

        # convert the image to greyscale and normalise it
        cur_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cur_img = cv2.undistort(cur_img, K0, dist_coeffs)
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        cur_img = torch.tensor(cur_img).unsqueeze(0).unsqueeze(0).float().to(device)

        # get the corresponding ground truth pose
        cur_pose = find_closest_timestamp(t.to_sec(), gt_poses)
        cur_quat = [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]
        cur_translation = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]
        cur_tf_mat = tf_trans.quaternion_matrix(cur_quat)
        cur_tf_mat[0:3, 3] = cur_translation
        cur_tf_mat = np.dot(T_rgb0_vlp16, cur_tf_mat)

        if prev_img is not None and prev_tf_mat is not None:
            # calculate the ground truth relative translation and rotation
            relative_gt_tf_mat = np.dot(np.linalg.inv(prev_tf_mat), cur_tf_mat)
            cumulative_gt_tf_mat = np.dot(cumulative_gt_tf_mat, relative_gt_tf_mat)

            est_translation, est_rotation = model(cur_img)
            est_translation, est_rotation = est_translation[0].detach().cpu().numpy(), est_rotation[0].detach().cpu().numpy()
            cumulative_est_tf_mat[:3, 3] = est_translation
            cumulative_est_tf_mat[:3, :3] = est_rotation

            print(f'est = {cumulative_est_tf_mat} \n')
            print(f'gt = {cumulative_gt_tf_mat} \n')

            # add the ground truth pose to the ground truth path
            gt_pose_stamped = PoseStamped()
            gt_pose_stamped.header.stamp = rospy.Time.from_sec(t.to_sec())
            gt_pose_stamped.header.frame_id = 'map'
            gt_pose_stamped.pose.position.x = cumulative_gt_tf_mat[0, 3]
            gt_pose_stamped.pose.position.y = cumulative_gt_tf_mat[1, 3]
            gt_pose_stamped.pose.position.z = cumulative_gt_tf_mat[2, 3]
            gt_quat_array = tf_trans.quaternion_from_matrix(cumulative_gt_tf_mat)
            gt_pose_stamped.pose.orientation.x = gt_quat_array[0]
            gt_pose_stamped.pose.orientation.y = gt_quat_array[1]
            gt_pose_stamped.pose.orientation.z = gt_quat_array[2]
            gt_pose_stamped.pose.orientation.w = gt_quat_array[3]
            gt_path.poses.append(gt_pose_stamped)

            # publish the ground truth path message
            gt_path_pub.publish(gt_path)

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

        # update the previous image, timestamp and pose
        prev_img = cur_img
        prev_tf_mat = cur_tf_mat
        prev_translation = cur_translation

if __name__ == '__main__':
    train()
