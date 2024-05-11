import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rosbag
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import tf.transformations as tf_trans
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# disable gradient calculations as not needed for inference
torch.set_grad_enabled(False)

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

# create directory to store results
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)

# define file paths
translation_loss_file = os.path.join(result_dir, 'translation_loss_test.txt')
rotation_loss_file = os.path.join(result_dir, 'rotation_loss_test.txt')
total_loss_file = os.path.join(result_dir, 'total_loss_test.txt')

# store average losses for each epoch
translation_losses_epoch = []
rotation_losses_epoch = []
total_losses_epoch = []

class PoseEstimationModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, translation_output_dim, rotation_output_dim):
        super(PoseEstimationModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim + 1, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers=num_layers)
        self.fc_translation = nn.Linear(hidden_dim, hidden_dim)
        self.fc_rotation = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out_translation = nn.Linear(hidden_dim, translation_output_dim)
        self.fc_out_rotation = nn.Linear(hidden_dim, rotation_output_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # initialise weights using Xavier initialisation
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        nn.init.xavier_uniform_(self.fc_translation.weight)
        nn.init.xavier_uniform_(self.fc_rotation.weight)
        nn.init.xavier_uniform_(self.fc_out_translation.weight)
        nn.init.xavier_uniform_(self.fc_out_rotation.weight)

    def forward(self, matches, confidence):
        # matches: (num_keypoints, 2)
        # confidence: (num_keypoints)

        # concatenate matches and confidence along the last dimension
        inputs = torch.cat((matches, confidence.unsqueeze(-1)), dim=-1)

        # apply linear layer to project inputs to hidden dimension
        embeddings = self.embedding_layer(inputs)

        # relu activation
        embeddings = torch.relu(embeddings)

        # attention
        attention_output, _ = self.attention(embeddings.float(), embeddings.float(), embeddings.float())

        # transformer encoder
        output = self.transformer_encoder(attention_output)

        # translation head
        translation = self.fc_translation(output.squeeze())

        # relu activation
        translation = torch.relu(translation)

        # rotation head
        rotation = self.fc_rotation(output.squeeze())

        # relu activation
        rotation = torch.relu(rotation)

        # output heads
        translation_out = self.fc_out_translation(translation)
        rotation_out = self.fc_out_rotation(rotation)

        return translation_out, rotation_out
    
# hyperparams
input_dim = 2  # dimensionality of the keypoints
num_heads = 8    # number of attention heads
num_layers = 6   # number of transformer layers
hidden_dim = 1024   # dimensionality of the hidden layer
translation_output_dim = 3 # dimensionality of translation output
rotation_output_dim = 4 # dimensionality of rotation output

model = PoseEstimationModel(input_dim, num_heads, num_layers, hidden_dim, translation_output_dim, rotation_output_dim)
# load the trained weights
model_path = os.path.join(result_dir, 'pose_estimation_model.pth')
model.load_state_dict(torch.load(model_path))

# move model to device and set to evaluation mode
model.to(device)
model.eval()

# define loss function
criterion = nn.MSELoss()

num_epochs = 1
    
# SuperPoint and SuperGlue hyperparams
model_config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 1024
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

# instantiate the SuperPoint and SuperGlue models
feature_matcher = Matching(model_config).to(device)
    
bag = rosbag.Bag('Datasets/BotanicGarden/1006_01_img10hz600p.bag')

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

for epoch in range(num_epochs):
    translation_losses = []
    rotation_losses = []
    total_losses = []

    # initialise previous image, timestamp, ground truth pose and transformation matrix
    prev_img = None
    prev_t = None
    prev_pose = None
    prev_tf_mat = None

    # loop through image data in rosbag
    for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw'])):
        # exit on ctrl + c
        if rospy.is_shutdown():
            break

        # convert the image to greyscale and normalise it
        cur_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        cur_img = torch.from_numpy(cur_img/255.).float()[None, None].to(device)

        # get the corresponding ground truth pose
        cur_pose = find_closest_timestamp(t.to_sec(), gt_poses)
        cur_quat = [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]
        cur_tf_mat = tf_trans.quaternion_matrix(cur_quat)
        cur_tf_mat[0:3, 3] = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]

        # if this is the first image or the time difference is greater than or equal to 1s
        if prev_t is None or t.to_sec() - prev_t.to_sec() >= 1:
            if prev_img is not None and prev_tf_mat is not None:
                # perform the matching
                pred = feature_matcher({'image0': prev_img, 'image1': cur_img})
                pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                matches, conf = pred['matches0'], pred['matching_scores0']
                
                # only keep the matched keypoints
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]

                # convert to tensors and move to device
                mkpts0_tensor = torch.from_numpy(mkpts0).float().to(device)
                mkpts1_tensor = torch.from_numpy(mkpts1).float().to(device)
                mconf_tensor = torch.from_numpy(mconf).float().to(device)

                # calculate the ground truth relative translation and rotation
                relative_tf_mat = np.dot(np.linalg.inv(prev_tf_mat), cur_tf_mat)
                translation_gt = relative_tf_mat[0:3, 3]
                rotation_gt = tf_trans.quaternion_from_matrix(relative_tf_mat)

                # convert np arrays to tensors and move to device
                translation_gt = torch.from_numpy(translation_gt).float().to(device)
                rotation_gt = torch.from_numpy(rotation_gt).float().to(device)

                # forward pass
                translation_pred, rotation_pred = model(mkpts0_tensor, mconf_tensor)

                # get predicted translations and rotations
                translation_pred, rotation_pred = translation_pred[1], rotation_pred[1]

                # compute the loss
                loss_translation = criterion(translation_pred, translation_gt)
                loss_rotation = criterion(rotation_pred, rotation_gt)
                loss = loss_translation + loss_rotation

                # append losses to lists
                translation_losses.append(loss_translation.item())
                rotation_losses.append(loss_rotation.item())
                total_losses.append(loss.item())

                translation_pred = translation_pred.cpu().numpy()
                rotation_pred = rotation_pred.cpu().numpy()

                # normalise rotation quaternion to ensure it is valid
                norm = np.linalg.norm(rotation_pred)
                rotation_pred /= norm

                # print(f' loss_translation = {loss_translation.item()}')
                # print(f' loss_rotation = {loss_rotation.item()}')
                # print(f' loss = {loss.item()} \n')

                # print(f'translation_pred = {translation_pred}')
                # print(f'translation_gt = {translation_gt}')
                # print(f'rotation_pred = {rotation_pred}')
                # print(f'rotation_gt = {rotation_gt} \n')

                # add the ground truth pose to the ground truth path
                gt_pose_stamped = PoseStamped()
                gt_pose_stamped.header.stamp = rospy.Time.from_sec(prev_t.to_sec())
                gt_pose_stamped.header.frame_id = 'map'
                gt_pose_stamped.pose = prev_pose
                gt_path.poses.append(gt_pose_stamped)

                # publish the ground truth path message
                gt_path_pub.publish(gt_path)
                # print(f'ground truth path updated')

                # add the estimated pose to the estimated path
                est_pose_stamped = PoseStamped()
                est_pose_stamped.header.stamp = rospy.Time.from_sec(t.to_sec())
                est_pose_stamped.header.frame_id = 'map'
                est_pose_stamped.pose.position.x = translation_pred[0]
                est_pose_stamped.pose.position.y = translation_pred[1]
                est_pose_stamped.pose.position.z = translation_pred[2]
                est_pose_stamped.pose.orientation.x = rotation_pred[0]
                est_pose_stamped.pose.orientation.y = rotation_pred[1]
                est_pose_stamped.pose.orientation.z = rotation_pred[2]
                est_pose_stamped.pose.orientation.w = rotation_pred[3]
                est_path.poses.append(est_pose_stamped)

                # publish the estimated path message
                est_path_pub.publish(est_path)
                # print(f'estimated path updated')

                print(f'gt_pose_stamped = {gt_pose_stamped}')
                print(f'est_pose_stamped = {est_pose_stamped}')

            # update the previous image, timestamp and pose
            prev_img = cur_img
            prev_t = t
            prev_pose = cur_pose
            prev_tf_mat = cur_tf_mat

    avg_translation_loss = sum(translation_losses) / len(translation_losses)
    avg_rotation_loss = sum(rotation_losses) / len(rotation_losses)
    avg_total_loss = sum(total_losses) / len(total_losses)

    # Store average losses for the epoch
    translation_losses_epoch.append(avg_translation_loss)
    rotation_losses_epoch.append(avg_rotation_loss)
    total_losses_epoch.append(avg_total_loss)

    # Save losses to files
    with open(translation_loss_file, 'a') as f:
        f.write(f'{avg_translation_loss}\n')

    with open(rotation_loss_file, 'a') as f:
        f.write(f'{avg_rotation_loss}\n')

    with open(total_loss_file, 'a') as f:
        f.write(f'{avg_total_loss}\n')

    print(f'epoch {epoch}: avg_translation_loss = {avg_translation_loss}')
    print(f'epoch {epoch}: avg_rotation_loss = {avg_rotation_loss}')
    print(f'epoch {epoch}: avg_total_loss = {avg_total_loss} \n')
