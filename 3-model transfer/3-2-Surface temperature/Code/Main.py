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

# Create file directory
folder_name = supply_function.create_file_directory()

# Model params
batch_size = 128
epochs = 50  #for demonstration
lr = 0.0001
train_loss = nn.MSELoss()

# Create dataset and split into training and validation sets
dataset = TensorDataset(data, targets)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define and train model
train_loss_list, val_loss_list, model = supply_function.model_define_train(train_loader, val_loader, epochs, lr, folder_name, train_loss)

# Plot training and validation losses
plt.figure()
plt.plot(range(1, epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(1, epochs + 1), val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
fig_path = os.path.join(folder_name, "loss_plot.png")
plt.savefig(fig_path)
plt.show()

# Model assessment
model.load_state_dict(torch.load(folder_name + "/train_try_1.pt"))
model.eval()
with torch.no_grad():
    full_outputs = model(data.unsqueeze(1)).squeeze().detach().numpy()

fig = plt.figure(figsize=(8, 6))
plt.plot(targets, label='Ground Truth')
plt.plot(full_outputs, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Output Value')
plt.legend()
fig_path = os.path.join(folder_name, "compare_plot.png")
plt.savefig(fig_path)
plt.show()
