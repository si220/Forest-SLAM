import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tf.transformations as tf_trans
import os
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device = {device}')

torch.manual_seed(0)

# create directory to store results
output_dir = 'training_results'
os.makedirs(output_dir, exist_ok=True)

# paths to files to store training and validation losses
training_losses = os.path.join(output_dir, 'training_losses.txt')
validation_losses = os.path.join(output_dir, 'validation_losses.txt')

class BotanicGarden(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = os.listdir(data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(file, allow_pickle=True).item()

        mkpts0 = torch.tensor(data['mkpts0']).float()
        mkpts1 = torch.tensor(data['mkpts1']).float()
        mconf = torch.tensor(data['mconf']).float()
        tf_mat = torch.tensor(data['tf_mat']).float()

        return mkpts0, mkpts1, mconf, tf_mat

class PoseEstimationModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super(PoseEstimationModel, self).__init__()
        self.embedding = nn.Linear(5, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.fc_translation = nn.Linear(d_model, 3)
        self.fc_rotation = nn.Linear(d_model, 4)

    def forward(self, x):
        x = x.transpose(0, 1)  # transformer expects seq_len, batch_size, embedding_dim
        x = self.embedding(x)
        output = self.transformer(x)
        output = output.transpose(0, 1)  # convert back to batch_size, seq_len, embedding_dim
        translation = self.fc_translation(output[:, 0, :])
        rotation = self.fc_rotation(output[:, 0, :])
        rotation = F.normalize(rotation, p=2, dim=-1)  # normalise the quaternion

        return translation, rotation
    
def train():
    model = PoseEstimationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable parameters = {num_params}')

    def collate(batch):
        mkpts0 = [item[0] for item in batch]
        mkpts1 = [item[1] for item in batch]
        mconf = [item[2] for item in batch]
        tf_mat = [item[3] for item in batch]

        # pad the sequences
        mkpts0 = pad_sequence(mkpts0, batch_first=True)
        mkpts1 = pad_sequence(mkpts1, batch_first=True)
        mconf = pad_sequence(mconf, batch_first=True)

        return mkpts0, mkpts1, mconf, tf_mat

    train_data_dir = 'Datasets/BotanicGarden/training_data/'
    val_data_dir = 'Datasets/BotanicGarden/validation_data/'

    # split dataset into training and validation sets
    train_set = BotanicGarden(train_data_dir)
    val_set = BotanicGarden(val_data_dir)

    # create training and validation dataloaders
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate)

    num_epochs = 30
    for epoch in range(num_epochs):
        # training phase
        model.train()
        training_loss = []

        for mkpts0, mkpts1, mconf, tf_mat_gt in train_dataloader:
            mkpts0, mkpts1, mconf = mkpts0.to(device), mkpts1.to(device), mconf.to(device)
            tf_mat_gt = torch.stack(tf_mat_gt).to(device)

            features = torch.cat((mkpts0, mkpts1, mconf.unsqueeze(-1)), dim=-1)

            # forward pass
            translation_pred, rotation_pred = model(features)

            translation_gt = tf_mat_gt[:, :3, 3]
            rotation_gt = torch.tensor([tf_trans.quaternion_from_matrix(R.cpu().numpy()) for R in tf_mat_gt], dtype=torch.float32).to(device)

            translation_loss = criterion(translation_pred, translation_gt)
            rotation_loss = criterion(rotation_pred, rotation_gt)
            loss = translation_loss + rotation_loss

            # backward pass and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append losses to lists
            training_loss.append(loss.item())

            print(f'translation_pred = {translation_pred[0].detach().cpu().numpy()}')
            print(f'translation_gt = {translation_gt[0].detach().cpu().numpy()}')
            print(f'rotation_pred = {rotation_pred[0].detach().cpu().numpy()}')
            print(f'rotation_gt = {rotation_gt[0].detach().cpu().numpy()} \n')

        avg_training_loss = sum(training_loss) / len(training_loss)
        print(f'epoch {epoch}: training loss = {avg_training_loss} \n')

        with open(training_losses, 'a') as f:
            f.write(str(avg_training_loss) + '\n')

        # validation phase
        model.eval()
        with torch.no_grad():
            validation_loss = []

            for mkpts0, mkpts1, mconf, tf_mat_gt in val_dataloader:
                mkpts0, mkpts1, mconf = mkpts0.to(device), mkpts1.to(device), mconf.to(device)
                tf_mat_gt = torch.stack(tf_mat_gt).to(device)

                features = torch.cat((mkpts0, mkpts1, mconf.unsqueeze(-1)), dim=-1)

                # forward pass
                translation_pred, rotation_pred = model(features)

                translation_gt = tf_mat_gt[:, :3, 3]
                rotation_gt = torch.tensor([tf_trans.quaternion_from_matrix(R.cpu().numpy()) for R in tf_mat_gt], dtype=torch.float32).to(device)

                translation_loss = criterion(translation_pred, translation_gt)
                rotation_loss = criterion(rotation_pred, rotation_gt)
                loss = translation_loss + rotation_loss

                # append losses to lists
                validation_loss.append(loss.item())

                print(f'translation_pred = {translation_pred[0].detach().cpu().numpy()}')
                print(f'translation_gt = {translation_gt[0].detach().cpu().numpy()}')
                print(f'rotation_pred = {rotation_pred[0].detach().cpu().numpy()}')
                print(f'rotation_gt = {rotation_gt[0].detach().cpu().numpy()} \n')

            avg_validation_loss = sum(validation_loss) / len(validation_loss)
            print(f'epoch {epoch}: validation loss = {avg_validation_loss} \n')

            with open(validation_losses, 'a') as f:
                f.write(str(avg_validation_loss) + '\n')

    # save the final trained model
    torch.save(model.state_dict(), os.path.join(output_dir, f'pose_estimation_model_{num_epochs}_epochs.pth'))

if __name__ == '__main__':
    train()
