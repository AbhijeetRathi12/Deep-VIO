
import torch.nn as nn
import sys
import torch

# Don't generate pyc codes
sys.dont_write_bytecode = True

device = 'cuda'

def LossFn(pred, target):
    
    loss_pos = nn.MSELoss()
    loss_poss = torch.sqrt(loss_pos(pred[:,:3], target[:,:3]))
    # ipdb.set_trace()
    loss_angle = nn.CosineEmbeddingLoss()
    loss_angles  = -loss_angle(pred[:,3:], target[:,3:], torch.ones(target[:,3:].shape[0], device=target[:,3:].device))
    loss = 0.4*loss_poss + 0.6*loss_angles
    return loss

def conv(in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Dropout(dropout)
    )

class Visual_encoder(nn.Module):
    def __init__(self):
        super(Visual_encoder, self).__init__()

        self.conv1 = conv(6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(256, 512, kernel_size=3, stride=2, dropout=0.2)
        
        # Define LSTM layer
        self.lstm = nn.LSTM(512, 256, 1, batch_first=True)
        # Define linear layer
        self.linear = nn.Linear(256, 6)

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv6 = self.conv4(out_conv3)
        return out_conv6

    def forward(self, x):
        
        x = self.encode_image(x)
        batch_size, channels, seq_len, variable = x.size() # N,L,H
        x = x.view(batch_size, seq_len * variable, channels)
        output, (_, _) = self.lstm(x)
        lstm_out = output[:, -1, :]
        output = self.linear(lstm_out)
        return output
    
    def validation_step(self, Img_test_batch, pose_test_batch):
        
        prediction = self.forward(Img_test_batch)
        loss_val = LossFn(prediction, pose_test_batch)
        return loss_val

class Inertial_encoder(nn.Module):
    def __init__(self):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.1))
        self.lstm = nn.LSTM(256, 64, 2, batch_first=True)

        self.linear = nn.Linear(64, 6)

    def forward(self, x):
        
        x = self.encoder_conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        out = self.linear(h_n[-1])
        return out

    def validation_step(self, IMU_test_batch, pose_test_batch):
        
        prediction = self.forward(IMU_test_batch)
        loss_val = LossFn(prediction, pose_test_batch)
        return loss_val
    
class Visual_Inertial_encoder(nn.Module):
    def __init__(self):
        super(Visual_Inertial_encoder, self).__init__()

        self.visual = Visual_encoder().to(device)
        self.inertial = Inertial_encoder().to(device)

        self.linear1 = nn.Conv1d(2, 64, kernel_size=3)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.linear2 = nn.Linear(256, 6)

    def forward(self, Img_train_batch, IMU_train_batch):
        
        img_pose = self.visual(Img_train_batch)
        inertial_pose = self.inertial(IMU_train_batch)
        x = torch.cat((img_pose.unsqueeze(1), inertial_pose.unsqueeze(1)), dim=1).to(device)
        x = self.linear1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        out = self.linear2(x)
        
        return out
    
    def validation_step(self, Img_test_batch, IMU_test_batch, pose_test_batch):
        
        prediction = self.forward(Img_test_batch, IMU_test_batch)
        loss_val = LossFn(prediction, pose_test_batch)
        return loss_val