import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from SuperGluePretrainedNetwork.models.matching import Matching
import matplotlib.pyplot as plt

# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

# set random seed for reproducibility
torch.manual_seed(0)

class PoseEstimationModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers):
        super(PoseEstimationModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for sequence aggregation
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # final linear layer to output 4x4 transformation matrix
        self.fc = nn.Linear(hidden_dim, 16)

    def forward(self, matched_kpts0, matched_kpts1):
        # concatenate matched keypoints into (batch_size, num_keypoints, 4)
        x = torch.cat((matched_kpts0, matched_kpts1), dim=-1)

        # LSTM input shape: (batch_size, sequence_length=num_keypoints, input_size=4)
        _, (h_n, _) = self.lstm(x)

        # get the final hidden state from LSTM
        h_n = h_n[-1]

        # final linear layer (batch_size, 16)
        pose_pred = self.fc(h_n)

        # reshape to 4x4 transformation matrix (batch_size, 4, 4)
        pose_pred = pose_pred.view(-1, 4, 4)
        
        return pose_pred

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

# instantiate the model with the same parameters used during training
input_size = 4
hidden_dim = 64
num_layers = 2

model = PoseEstimationModel(input_size=input_size, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

# load the trained model state dict
model_path = 'Forest_SLAM_BotanicGarden_1006_01.pth'
model.load_state_dict(torch.load(model_path))

# set the model to evaluation mode
model.eval()

class BotanicGardenTestDataset(Dataset):
    def __init__(self, root_dir, poses_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(poses_file, 'r') as f:
            self.poses = f.readlines()

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        line = self.poses[idx].strip().split()
        img0_path = os.path.join(self.root_dir, line[0])
        img1_path = os.path.join(self.root_dir, line[1])
        pose = np.array(line[22:], dtype=np.float32).reshape(4, 4)

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, pose

transform = transforms.Compose([
    transforms.ToTensor()
])

# instantiate the test dataset and dataloader
test_dataset = BotanicGardenTestDataset(root_dir='Datasets/BotanicGarden/1018_00/greyscale_imgs/',
                                        poses_file='Datasets/BotanicGarden/1018_00/poses.txt',
                                        transform=transform)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

test_losses = []

# testing loop
for i, (img0, img1, pose) in enumerate(test_dataloader):
    img0 = img0.to(device)
    img1 = img1.to(device)
    pose = pose.to(device)

    # SuperGlue inference
    output = feature_matcher({'image0': img0, 'image1': img1})
    keypoints0 = output['keypoints0'][0]
    keypoints1 = output['keypoints1'][0]
    num_matches = output['matches0'][0]

    # keep only the valid matches
    correct = num_matches > -1
    matched_kpts0 = keypoints0[correct]
    matched_kpts1 = keypoints1[num_matches[correct]]

    # remove gradient computation since the model isn't being trained
    with torch.no_grad():
        pose_pred = model(matched_kpts0, matched_kpts1)

    # compute the loss
    rotation_loss = torch.norm(pose_pred[:, :3, :3] - pose[:, :3, :3])
    translation_loss = torch.norm(pose_pred[:, :3, 3] - pose[:, :3, 3])
    total_loss = rotation_loss + translation_loss

    # track loss
    test_losses.append(total_loss.item())

    print(f'Test Sample {i+1}, Loss: {total_loss.item()}')

# compute average test loss
avg_test_loss = sum(test_losses) / len(test_losses)
print(f'Average Test Loss: {avg_test_loss}')

# plot test losses
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Test Sample')
plt.ylabel('Loss')
plt.title('Test Loss per Sample')
plt.legend()
plt.grid(True)
plt.show()
