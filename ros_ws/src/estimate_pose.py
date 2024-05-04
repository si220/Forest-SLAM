import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import rosbag
import tf.transformations as tf_trans
import cv2
from cv_bridge import CvBridge
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, translation_output_dim, rotation_output_dim):
        super(TransformerModel, self).__init__()
        self.embedding_layer = nn.Linear(input_dim + 1, hidden_dim)  # Concatenate matches and confidence
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers=num_layers)
        self.fc_translation = nn.Linear(hidden_dim, hidden_dim)
        self.fc_rotation = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out_translation = nn.Linear(hidden_dim, translation_output_dim)
        self.fc_out_rotation = nn.Linear(hidden_dim, rotation_output_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, matches, confidence):
        # matches: (max_keypoints, 2)
        # confidence: (max_keypoints,)

        # Convert matches tensor to float
        matches = matches.float()

        # Concatenate matches and confidence along the last dimension
        inputs = torch.cat((matches, confidence.unsqueeze(-1)), dim=-1)

        # Apply linear layer to project inputs to hidden dimension
        embeddings = self.embedding_layer(inputs)

        # Reshape embeddings to match transformer input shape
        embeddings = embeddings.unsqueeze(1).transpose(0, 1)  # (1, max_keypoints, hidden_dim)

        # Apply attention using transformer encoder
        attention_output, _ = self.attention(embeddings.float(), embeddings.float(), embeddings.float())

        # Transformer encoder
        output = self.transformer_encoder(attention_output)

        # Translation head
        translation = self.fc_translation(output.squeeze())

        # Rotation head
        rotation = self.fc_rotation(output.squeeze())

        # Output prediction
        translation_out = self.fc_out_translation(translation)
        rotation_out = self.fc_out_rotation(rotation)

        return translation_out, rotation_out

class PoseDataset(Dataset):
    def __init__(self, bag_file, image_topic, pose_topic):
        self.bag = rosbag.Bag(bag_file)
        self.image_topic = image_topic
        self.pose_topic = pose_topic
        self.bridge = CvBridge()
        self.image_msgs = self.bag.read_messages(topics=[self.image_topic])
        self.pose_msgs = self.bag.read_messages(topics=[self.pose_topic])
        self.len = self.bag.get_message_count(topic_filters=[self.image_topic]) - 1
        # Initialize pose_msg0 and pose_msg1
        self.pose_msg0 = next(self.pose_msgs, None)
        self.pose_msg1 = next(self.pose_msgs, None)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Get the idx-th image message and the next one
        img_msg0 = next(self.image_msgs)
        img_msg1 = next(self.image_msgs)
        # Convert the images to OpenCV format
        img0 = self.bridge.imgmsg_to_cv2(img_msg0.message, desired_encoding='passthrough')
        img1 = self.bridge.imgmsg_to_cv2(img_msg1.message, desired_encoding='passthrough')
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # convert to tensors
        img0 = torch.from_numpy(img0/255.).float()[None, None].to(device)
        img1 = torch.from_numpy(img1/255.).float()[None, None].to(device)

        # Get the corresponding pose messages
        while self.pose_msg1 is not None and self.pose_msg1.timestamp < img_msg1.timestamp:
            self.pose_msg0 = self.pose_msg1
            self.pose_msg1 = next(self.pose_msgs, None)

        # Convert the PoseStamped messages to arrays
        pose0 = np.array([self.pose_msg0.message.pose.position.x, self.pose_msg0.message.pose.position.y, self.pose_msg0.message.pose.position.z,
                          self.pose_msg0.message.pose.orientation.x, self.pose_msg0.message.pose.orientation.y,
                          self.pose_msg0.message.pose.orientation.z, self.pose_msg0.message.pose.orientation.w])
        pose1 = np.array([self.pose_msg1.message.pose.position.x, self.pose_msg1.message.pose.position.y, self.pose_msg1.message.pose.position.z,
                          self.pose_msg1.message.pose.orientation.x, self.pose_msg1.message.pose.orientation.y,
                          self.pose_msg1.message.pose.orientation.z, self.pose_msg1.message.pose.orientation.w])

        # Return the data
        return img0, img1, pose0, pose1

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

# Instantiate the SuperPoint and SuperGlue models
feature_matcher = Matching(model_config).to(device)

bag_file = 'Datasets/BotanicGarden/1006_01_img10hz600p.bag'
image_topic = '/dalsa_rgb/left/image_raw'
pose_topic = '/gt_poses'

# Create dataset and dataloader
dataset = PoseDataset(bag_file, image_topic, pose_topic)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Define Transformer model
input_dim = 2  # Dimensionality of the keypoints
num_heads = 8    # Number of attention heads
num_layers = 6   # Number of transformer layers
hidden_dim = 512 # Dimensionality of the hidden layer
translation_output_dim = 3 # Dimensionality of rotation output
rotation_output_dim = 6 # Dimensionality of rotation output

model = TransformerModel(input_dim, num_heads, num_layers, hidden_dim, translation_output_dim, rotation_output_dim)
model.to(device)
model.train()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        # skip first iteration since there are no image pairs
        if batch_idx == 0:
            continue

        # get pairs of adjacent images along with their ground truth poses
        img0, img1, pose0, pose1 = data

        # remove extra dimension
        img0 = img0.squeeze(0)
        img1 = img1.squeeze(0)
        pose0 = pose0.squeeze(0)
        pose1 = pose1.squeeze(0)

        # Perform the matching.
        pred = feature_matcher({'image0': img0, 'image1': img1})
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Convert to tensors and move to device
        mkpts0_tensor = torch.from_numpy(mkpts0).float().to(device)
        mkpts1_tensor = torch.from_numpy(mkpts1).float().to(device)
        mconf_tensor = torch.from_numpy(mconf).float().to(device)

        translation0 = pose0[:3]
        translation1 = pose1[:3]

        rotation0 = tf_trans.quaternion_matrix(pose0[3:])[0:3, 0:3]
        rotation1 = tf_trans.quaternion_matrix(pose1[3:])[0:3, 0:3]

        # Compute the relative translation and rotation in quaternion form
        translation_gt = np.dot(rotation0.T, translation1 - translation0)
        rotation_gt_quat = tf_trans.quaternion_multiply(tf_trans.quaternion_inverse(pose0[3:]), pose1[3:])

        # Convert the relative rotation from quaternion to 6D representation
        rotation_gt_mat = tf_trans.quaternion_matrix(rotation_gt_quat)[:3, :3]
        rotation_gt_6D = rotation_gt_mat[:, :2].reshape(-1)

        # Convert np arrays to tensors and move to device
        translation_gt = torch.from_numpy(translation_gt).float().to(device)
        rotation_gt = torch.from_numpy(rotation_gt_6D).float().to(device)

        # Forward pass
        translation_pred, rotation_pred = model(mkpts0_tensor, mconf_tensor)

        translation_pred = translation_pred[1]
        rotation_pred = rotation_pred[1]

        # Compute the loss
        loss_translation = criterion(translation_pred, translation_gt)
        loss_rotation = criterion(rotation_pred, rotation_gt)
        loss = loss_translation + loss_rotation

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        print(f'predicted translation = {translation_pred}')
        print(f'actual translation = {translation_gt}')

        # print(f'pose loss = {loss.item()}')

    # Print epoch statistics
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss / len(dataloader)))
