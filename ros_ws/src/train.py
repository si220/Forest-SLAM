import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rosbag
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import tf.transformations as tf_trans
import cv2
import numpy as np
import matplotlib.pyplot as plt
from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

bridge = CvBridge()

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
model.to(device)
model.train()

# define loss function and optimiser
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10
    
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

# initialise previous image, timestamp and pose
prev_img = None
prev_t = None
prev_tf_mat = None

for epoch in range(num_epochs):
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

                # backward pass and optimisation
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                print(f' loss_translation = {loss_translation.item()}')
                print(f' loss_rotation = {loss_rotation.item()}')
                print(f' loss = {loss.item()} \n')

                print(f'translation_pred = {translation_pred}')
                print(f'translation_gt = {translation_gt}')
                print(f'rotation_pred = {rotation_pred}')
                print(f'rotation_gt = {rotation_gt} \n')

            # update the previous image, timestamp and pose
            prev_img = cur_img
            prev_t = t
            prev_tf_mat = cur_tf_mat

