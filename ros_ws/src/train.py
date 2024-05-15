import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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

torch.manual_seed(0)

bridge = CvBridge()

# create directory to store results
result_dir = 'results'
os.makedirs(result_dir, exist_ok=True)

# define file paths
translation_loss_file = os.path.join(result_dir, 'translation_loss_train.txt')
rotation_loss_file = os.path.join(result_dir, 'rotation_loss_train.txt')
total_loss_file = os.path.join(result_dir, 'total_loss_train.txt')

# store average losses for each epoch
translation_losses_epoch = []
rotation_losses_epoch = []
total_losses_epoch = []

def my_collate(batch):
    matches = pad_sequence([item[0] for item in batch], batch_first=True)
    confidence = pad_sequence([item[1] for item in batch], batch_first=True)
    translation_gt = pad_sequence([item[2] for item in batch], batch_first=True)
    rotation_gt = pad_sequence([item[3] for item in batch], batch_first=True)

    return matches, confidence, translation_gt, rotation_gt

class BotanicGarden(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = os.listdir(data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(file, allow_pickle=True).item()

        matches = torch.tensor(data['matches']).float()
        confidence = torch.tensor(data['confidence']).float()
        translation_gt = torch.tensor(data['translation_gt']).float()
        rotation_gt = torch.tensor(data['rotation_gt']).float()

        return matches, confidence, translation_gt, rotation_gt


class PoseEstimationModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, translation_output_dim, rotation_output_dim, dropout_prob):
        super(PoseEstimationModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim + 1, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers=num_layers)
        self.fc_translation = nn.Linear(hidden_dim, 2 * hidden_dim)  # Increased output dimension
        self.fc_rotation = nn.Linear(hidden_dim, 2 * hidden_dim)     # Increased output dimension
        self.fc_out_translation = nn.Linear(2 * hidden_dim, translation_output_dim)  # Adjusted output dimension
        self.fc_out_rotation = nn.Linear(2 * hidden_dim, rotation_output_dim)        # Adjusted output dimension
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        nn.init.xavier_uniform_(self.fc_translation.weight)
        nn.init.xavier_uniform_(self.fc_rotation.weight)
        nn.init.xavier_uniform_(self.fc_out_translation.weight)
        nn.init.xavier_uniform_(self.fc_out_rotation.weight)

    def forward(self, matches, confidence):
        # matches: (num_keypoints, 2)
        # confidence: (num_keypoints)

        # Concatenate matches and confidence along the last dimension
        inputs = torch.cat((matches, confidence.unsqueeze(-1)), dim=-1)

        # Apply linear layer to project inputs to hidden dimension
        embeddings = self.embedding_layer(inputs)

        # ReLU activation
        embeddings = torch.relu(embeddings)

        # Attention
        attention_output, _ = self.attention(embeddings.float(), embeddings.float(), embeddings.float())

        # Transformer encoder
        output = self.transformer_encoder(attention_output)

        # Translation head
        translation = self.fc_translation(output.squeeze())

        # ReLU activation
        translation = torch.relu(translation)

        # Rotation head
        rotation = self.fc_rotation(output.squeeze())

        # ReLU activation
        rotation = torch.relu(rotation)

        # Apply dropout
        translation = self.dropout(translation)
        rotation = self.dropout(rotation)

        # Output heads
        translation_out = self.fc_out_translation(translation)
        rotation_out = self.fc_out_rotation(rotation)

        # Normalize quaternion predictions using L2 norm
        rotation_out = F.normalize(rotation_out, p=2, dim=-1)

        return translation_out, rotation_out
    
# hyperparams
input_dim = 2  # dimensionality of the keypoints
num_heads = 8    # number of attention heads
num_layers = 6   # number of transformer layers
hidden_dim = 1024   # dimensionality of the hidden layer
translation_output_dim = 3 # dimensionality of translation output
rotation_output_dim = 4 # dimensionality of rotation output
dropout_prob = 0.1 # probability of dropping out random neuron

model = PoseEstimationModel(input_dim, num_heads, num_layers, hidden_dim, translation_output_dim, rotation_output_dim, dropout_prob)
model.to(device)
model.train()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'trainable parameters = {num_params}')

# define loss function and optimiser
criterion = nn.SmoothL1Loss()  # Huber Loss
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

num_epochs = 50

data_dir = 'Datasets/Pose_Estimation_Data/'

# create dataset and dataloader
dataset = BotanicGarden(data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=my_collate)

for epoch in range(num_epochs):
    translation_losses = []
    rotation_losses = []
    total_losses = []

    for matches, confidence, translation_gt, rotation_gt in dataloader:
        matches, confidence = matches.to(device), confidence.to(device)
        translation_gt, rotation_gt = translation_gt.to(device), rotation_gt.to(device)

        # forward pass
        translation_pred, rotation_pred = model(matches, confidence)

        # shape = [batch size, 3]
        translation_pred = translation_pred[:, -1, :]
        # shape = [batch size, 4]
        rotation_pred = rotation_pred[:, -1, :]

        # compute the loss
        loss_translation = criterion(translation_pred, translation_gt)
        loss_rotation = criterion(rotation_pred, rotation_gt)
        loss = loss_translation + loss_rotation

        # backward pass and optimisation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # append losses to lists
        translation_losses.append(loss_translation.item())
        rotation_losses.append(loss_rotation.item())
        total_losses.append(loss.item())

        # print(f' loss_translation = {loss_translation.item()}')
        # print(f' loss_rotation = {loss_rotation.item()}')
        # print(f' loss = {loss.item()} \n')

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

    
# bag = rosbag.Bag('Datasets/BotanicGarden/1006_01_img10hz600p.bag')

# # create a dictionary to store the ground truth poses
# gt_poses = {}

# # loop through ground truth poses in rosbag
# for topic, msg, t in bag.read_messages(topics=['/gt_poses']):
#     gt_poses[t.to_sec()] = msg.pose

# # function to find current pose based on closest timestamp
# def find_closest_timestamp(target, timestamp_dict):
#     # get the list of timestamps
#     timestamps = np.array(list(timestamp_dict.keys()))

#     # find the index of the closest timestamp
#     idx = np.argmin(np.abs(timestamps - target))

#     # return the pose corresponding to the closest timestamp
#     return timestamp_dict[timestamps[idx]]

# for epoch in range(num_epochs):
#     translation_losses = []
#     rotation_losses = []
#     total_losses = []

#     # initialise previous image, timestamp, ground truth pose and transformation matrix
#     prev_img = None
#     prev_t = None
#     prev_pose = None
#     prev_tf_mat = None

#     # loop through image data in rosbag
#     for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/dalsa_rgb/left/image_raw'])):
#         # exit on ctrl + c
#         if rospy.is_shutdown():
#             break

#         # convert the image to greyscale and normalise it
#         cur_img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
#         cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
#         cur_img = torch.from_numpy(cur_img/255.).float()[None, None].to(device)

#         # get the corresponding ground truth pose
#         cur_pose = find_closest_timestamp(t.to_sec(), gt_poses)
#         cur_quat = [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]
#         cur_tf_mat = tf_trans.quaternion_matrix(cur_quat)
#         cur_tf_mat[0:3, 3] = [cur_pose.position.x, cur_pose.position.y, cur_pose.position.z]

#         # if this is the first image or the time difference is greater than or equal to 1s
#         if prev_t is None or t.to_sec() - prev_t.to_sec() >= 0.5:
#             if prev_img is not None and prev_tf_mat is not None:
#                 # perform the matching
#                 pred = feature_matcher({'image0': prev_img, 'image1': cur_img})
#                 pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
#                 kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
#                 matches, conf = pred['matches0'], pred['matching_scores0']
                
#                 # only keep the matched keypoints
#                 valid = matches > -1
#                 mkpts0 = kpts0[valid]
#                 mkpts1 = kpts1[matches[valid]]
#                 mconf = conf[valid]

#                 # convert to tensors and move to device
#                 mkpts0_tensor = torch.from_numpy(mkpts0).float().to(device)
#                 mkpts1_tensor = torch.from_numpy(mkpts1).float().to(device)
#                 mconf_tensor = torch.from_numpy(mconf).float().to(device)

#                 # calculate the ground truth relative translation and rotation
#                 relative_tf_mat = np.dot(np.linalg.inv(prev_tf_mat), cur_tf_mat)
#                 translation_gt = relative_tf_mat[0:3, 3]
#                 rotation_gt = tf_trans.quaternion_from_matrix(relative_tf_mat)

#                 # convert np arrays to tensors and move to device
#                 translation_gt = torch.from_numpy(translation_gt).float().to(device)
#                 rotation_gt = torch.from_numpy(rotation_gt).float().to(device)

#                 # forward pass
#                 translation_pred, rotation_pred = model(mkpts0_tensor, mconf_tensor)

#                 # get predicted translations and rotations
#                 translation_pred, rotation_pred = translation_pred[1], rotation_pred[1]

#                 # compute the loss
#                 loss_translation = criterion(translation_pred, translation_gt)
#                 loss_rotation = criterion(rotation_pred, rotation_gt)
#                 loss = loss_translation + loss_rotation

#                 # backward pass and optimisation
#                 optimiser.zero_grad()
#                 loss.backward()
#                 optimiser.step()

#                 # append losses to lists
#                 translation_losses.append(loss_translation.item())
#                 rotation_losses.append(loss_rotation.item())
#                 total_losses.append(loss.item())

#                 print(f' loss_translation = {loss_translation.item()}')
#                 print(f' loss_rotation = {loss_rotation.item()}')
#                 print(f' loss = {loss.item()} \n')

#                 print(f'translation_pred = {translation_pred}')
#                 print(f'translation_gt = {translation_gt}')
#                 print(f'rotation_pred = {rotation_pred}')
#                 print(f'rotation_gt = {rotation_gt} \n')

#             # update the previous image, timestamp and pose
#             prev_img = cur_img
#             prev_t = t
#             prev_pose = cur_pose
#             prev_tf_mat = cur_tf_mat

#     avg_translation_loss = sum(translation_losses) / len(translation_losses)
#     avg_rotation_loss = sum(rotation_losses) / len(rotation_losses)
#     avg_total_loss = sum(total_losses) / len(total_losses)

#     # Store average losses for the epoch
#     translation_losses_epoch.append(avg_translation_loss)
#     rotation_losses_epoch.append(avg_rotation_loss)
#     total_losses_epoch.append(avg_total_loss)

#     # Save losses to files
#     with open(translation_loss_file, 'a') as f:
#         f.write(f'{avg_translation_loss}\n')

#     with open(rotation_loss_file, 'a') as f:
#         f.write(f'{avg_rotation_loss}\n')

#     with open(total_loss_file, 'a') as f:
#         f.write(f'{avg_total_loss}\n')

#     print(f'epoch {epoch}: avg_translation_loss = {avg_translation_loss}')
#     print(f'epoch {epoch}: avg_rotation_loss = {avg_rotation_loss}')
#     print(f'epoch {epoch}: avg_total_loss = {avg_total_loss} \n')

# save the final trained model
torch.save(model.state_dict(), os.path.join(result_dir, f'pose_estimation_model_{num_epochs}_epochs.pth'))
