import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import supply_function
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Data preparation
data, targets = supply_function.training_data_preparation()


# Define model
class CNN2DModel(nn.Module):
    def __init__(self):
        super(CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)  # First dropout layer
        self.fc1 = nn.Linear(in_features=200, out_features=100)
        self.dropout2 = nn.Dropout(p=0.2)  # Second dropout layer
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Initialize the model, loss function, and optimizer
model = CNN2DModel()

# Model assessment
model.load_state_dict(torch.load('../Model' + "/Large_in.pt"))
model.eval()
with torch.no_grad():
    full_outputs = model(data.unsqueeze(1)).squeeze().detach().numpy()

fig = plt.figure(figsize=(20, 4))
plt.plot(targets, label='Ground Truth')
plt.plot(full_outputs, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Output Value')
plt.legend()
plt.show()
