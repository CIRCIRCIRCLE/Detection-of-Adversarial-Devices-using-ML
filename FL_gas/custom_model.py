import torch
import torch.nn as nn
import torch.nn.functional as F
from easyfl.models import BaseModel

class CustomizedLSTMCNN(BaseModel):
    def __init__(self, input_dim, num_classes):
        super(CustomizedLSTMCNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, 50, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 1, 64)  # Adjusted input size calculation for the FC layer based on input_dim
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ensure input tensor has 3 dimensions (batch_size, sequence_length, input_dim)
        #print(f'Input shape: {x.shape}')
        h_lstm, _ = self.lstm(x)
        #print(f'LSTM output shape: {h_lstm.shape}')
        h_lstm = h_lstm.permute(0, 2, 1)  # (batch_size, num_features, sequence_length)
        #print(f'Permuted LSTM shape: {h_lstm.shape}')
        h_conv = F.relu(self.conv1(h_lstm))
        #print(f'Conv1D output shape: {h_conv.shape}')
        h_flat = self.flatten(h_conv)
        #print(f'Flatten output shape: {h_flat.shape}')
        h_fc1 = F.relu(self.fc1(h_flat))
        out = self.fc2(h_fc1)
        return out



'''
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ensure input tensor has 3 dimensions (batch_size, sequence_length, input_dim)
        h_lstm, _ = self.lstm(x)
        h_lstm = h_lstm.permute(0, 2, 1)  # (batch_size, num_features, sequence_length)
        h_conv = F.relu(self.conv1(h_lstm))
        h_pool = self.pool(h_conv)
        h_flat = self.flatten(h_pool)
        h_fc1 = F.relu(self.fc1(h_flat))
        out = self.fc2(h_fc1)
        return out
'''