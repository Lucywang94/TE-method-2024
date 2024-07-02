import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Dataset generation function
def dataset_generation_function(DSC_name, ARC_name, Dataset_length, DSC_num, ARC_index=None):
    location = '../data/'
    # Load DSC data
    filename_DSC = f"{DSC_name}_{DSC_num}.txt"
    DSC_data = np.loadtxt(location + filename_DSC, skiprows=1,delimiter = ',')
    Temperature_DSC = DSC_data[:, 0]

    # Load ARC data
    filename_ARC = f"{ARC_name}.txt"
    data = np.loadtxt(location + filename_ARC, skiprows=1,delimiter = ',')
    Time_ARC = data[:, 0]
    Temperature_ARC = data[:, 1]
    dTdt_ARC = data[:, 2]

    # Define calculation range
    Temp_ARC_l = np.ceil(np.min(Temperature_ARC))
    Temp_ARC_h = np.floor(np.max(Temperature_ARC))
    Temp_DSC_l = np.ceil(np.min(Temperature_DSC))
    Temp_DSC_h = np.floor(np.max(Temperature_DSC))

    if Temp_ARC_l < 90:
        Temp_l = 90
    else:
        Temp_l = Temp_ARC_l

    Temp_h = min(Temp_ARC_h, Temp_DSC_h)

    # Calculate Temp_range and ARC_index,
    delta_SP = 0.005  #for demonstration
    Temp_range = np.arange(Temp_l, Temp_h + 0.0005, delta_SP).round(3)

    # Generate DSC_dataset
    DSC_dataset = np.zeros((len(Temp_range), Dataset_length, DSC_num+1))
    for i, temp in enumerate(Temp_range):
        Temp_i = np.linspace(Temp_DSC_l, temp, Dataset_length)
        interpolator = interp1d(Temperature_DSC, DSC_data[:, 1:], axis=0, fill_value="extrapolate")
        interpolated_data = interpolator(Temp_i)
        DSC_dataset[i, :, :1] = Temp_i[:, np.newaxis]
        DSC_dataset[i, :, 1:] = interpolated_data

    interpolatorARC = interp1d(Temperature_ARC, dTdt_ARC, fill_value="extrapolate")
    ARC_dTdt = interpolatorARC(Temp_range)
    log_ARC_dTdt=np.log(ARC_dTdt)

    #normalize Temperature within (50,1000)
    DSC_dataset_norm=np.copy(DSC_dataset)
    DSC_dataset_norm[:, :, 0]=(DSC_dataset_norm[:, :, 0]-50)/1000

    return DSC_dataset_norm, log_ARC_dTdt

# Data preparation
def training_data_preparation():
    DSC_num = 5
    Dataset_length = 100
    name_list = ["Si10_BOL", "Si10_EOL", "Si20_BOL", "Si20_EOL"]
    DSC_name_list = [f"DSC_{name}" for name in name_list]
    ARC_name_list = [f"ARC_{name}" for name in name_list]

    DSC_dataset_all = []
    ARC_dTdt_dataset_all = []

    for DSC_name, ARC_name in zip(DSC_name_list, ARC_name_list):
        DSC_dataset, log_ARC_dTdt_dataset = dataset_generation_function(DSC_name, ARC_name, Dataset_length, DSC_num)
        DSC_dataset_all.append(DSC_dataset)
        ARC_dTdt_dataset_all.append(log_ARC_dTdt_dataset)

    DSC_dataset_all = np.concatenate(DSC_dataset_all, axis=0)
    ARC_dTdt_dataset_all = np.concatenate(ARC_dTdt_dataset_all, axis=0)

    # Convert numpy arrays to PyTorch tensors
    data = torch.tensor(DSC_dataset_all, dtype=torch.float32)
    targets = torch.tensor(ARC_dTdt_dataset_all, dtype=torch.float32)
    return data, targets

def create_file_directory():
    # Create file directory
    today = datetime.datetime.now().strftime("%Y%m%d")
    folder_suffix = 1
    folder_name = f"result_try_{today}_{folder_suffix}"

    while os.path.exists(folder_name):
        folder_suffix += 1
        folder_name = f"result_try_{today}_{folder_suffix}"

    os.makedirs(folder_name)

    return folder_name

def model_define_train(train_loader, val_loader, epochs, lr, folder_name,train_loss):
    # Load pre-trained model parameters
    pretrained_params = torch.load('Pretrain_1Ah.pt')
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
            x = self.dropout1(torch.relu(self.fc1(x)))  # Dropout applied after first dense layer
            x = self.dropout2(torch.relu(self.fc2(x)))  # Dropout applied after second dense layer
            x = self.fc3(x)
            return x

    # Initialize the model, loss function, and optimizer
    model = CNN2DModel()

    # Load the pre-trained parameters
    model.load_state_dict(pretrained_params)

    # Freeze
    for name, param in model.named_parameters():
        if 'conv1' in name or 'fc2' in name:
            param.requires_grad = False

    # Fine-tune
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_loss_list = []
    val_loss_list = []

    # Training loop
    for epoch in range(epochs):
        train_loss_epoch = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1).float())
            loss = train_loss(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= len(train_loader)
        train_loss_list.append(train_loss_epoch)

        # Validation
        val_loss_epoch = 0.0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader, 0):
                outputs = model(inputs.unsqueeze(1).float())
                loss = train_loss(outputs, labels.float().unsqueeze(1))
                val_loss_epoch += loss.item()
            val_loss_epoch /= len(val_loader)
        val_loss_list.append(val_loss_epoch)

        print(f"Epoch {epoch + 1} - train_loss: {train_loss_epoch:.5f} - val_loss: {val_loss_epoch:.5f}")

        # Save model params
        def get_next_filename(folder_path, filename_prefix):
            existing_files = [f for f in os.listdir(folder_path) if
                              os.path.isfile(os.path.join(folder_path, f)) and f.startswith(filename_prefix)]
            max_number = 0
            for filename in existing_files:
                number = int(filename.split("_")[-1].split(".")[0])
                max_number = max(max_number, number)
            return f"{filename_prefix}_{max_number + 1}.pt"

        model_filename = get_next_filename(folder_name, "train_try")
        model_path = os.path.join(folder_name, model_filename)

    torch.save(model.state_dict(), model_path)
    print("Training complete.")
    return train_loss_list, val_loss_list, model