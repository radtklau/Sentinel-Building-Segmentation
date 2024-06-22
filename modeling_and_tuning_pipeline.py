import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm
from datetime import datetime
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, path_to_ds, mode):
        if mode not in ["train", "test", "val"]:
            print("Invalid mode.")
            sys.exit()
        else:
            features_fn = f"features_{mode}.npy"
            labels_fn = f"labels_{mode}.npy"

        features_fp = os.path.join(path_to_ds, features_fn)
        labels_fp = os.path.join(path_to_ds, labels_fn)

        if not (os.path.exists(features_fp) and os.path.exists(labels_fp)):
            raise FileNotFoundError(f"Data files not found for mode '{mode}'")

        self.features = np.load(features_fp)
        self.labels = np.load(labels_fp)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, ind):
        image = self.features[ind]
        label = self.labels[ind]
        return image, label

class PixelClassifier(nn.Module):
    def __init__(self):
        super(PixelClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = torch.sigmoid(x)
        return x

    
def get_dataloaders(path_to_ds, batch_size=32):
    print("Creating train, test and val datasets...")
    train_dataset = ImageDataset(path_to_ds, 'train')
    val_dataset = ImageDataset(path_to_ds, 'val')
    test_dataset = ImageDataset(path_to_ds, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    models_dir_name = "models"
    if not os.path.exists(models_dir_name):
        os.makedirs(models_dir_name)

    run_ind = 0
    this_run_dir_name = ""
    while True:
        path_to_run = f"models/run_{run_ind}"
        if not os.path.exists(path_to_run):
            this_run_dir_name = f"run_{run_ind}"
            break
        else:
            run_ind += 1

    this_run_dir_path = os.path.join(models_dir_name, this_run_dir_name)
    os.makedirs(this_run_dir_path)

    print("Training model...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    
    print("Saving model...")
    save_model(model, num_epochs, learning_rate, this_run_dir_path)
    return model

def save_model(model, num_epochs, learning_rate, this_run_dir_path):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"baseline_ds{path_to_ds[-1]}_ep{num_epochs}_lr{learning_rate}_bs{batch_size}_{formatted_datetime}.pth"
    path_to_model = os.path.join(this_run_dir_path, model_name)
    torch.save(model, path_to_model)

def evaluate_model(model, data_loader, criterion):
    print("Evaluating model...")
    model.eval()
    loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss += loss.item()
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
    
    loss /= len(data_loader)
    acc = accuracy_score(np.array(all_labels), np.array(all_preds))
    
    return loss, acc

def test_model(model_path):
    model = torch.load(model_path)
    path_to_label_image = "building_and_sentinel_data/Berlin_test/Berlin_test_buildings.png"
    path_to_rgb_image = "building_and_sentinel_data/Berlin_test/Berlin_test_rgb.png"
    label_im = Image.open(path_to_label_image).convert('RGB')
    label_im = np.array(label_im)

    building_mask = (label_im == [0, 0, 255]).all(axis=2) 
    label_matrix = np.zeros((label_im.shape[0], label_im.shape[1]), dtype=np.uint8)
    label_matrix[building_mask] = 1

    rgb_im = Image.open(path_to_rgb_image).convert('RGB')
    rgb_im = np.array(rgb_im)
    rgb_im = rgb_im.astype(np.float32) / 255.0
    rgb_im = np.expand_dims(rgb_im, axis=0)
    rgb_im = np.transpose(rgb_im, (0, 3, 1, 2))

    inp = torch.tensor(rgb_im)

    pred = model(inp)

    pred_matrix = (pred > 0.5).float()
    pred_matrix = pred_matrix.numpy().astype(np.uint8)[0, 0]

    acc = custom_eval(label_matrix, pred_matrix)

    print(f"{acc}%")

    plt.figure(figsize=(10, 5))
    
    # Plot the second image
    plt.subplot(1, 2, 1)
    plt.imshow(label_matrix, cmap='gray', vmin=0, vmax=1) 
    plt.title('labels')
    plt.axis('off')  # Optional: turn off axis labels

    # Plot the second image
    plt.subplot(1, 2, 2)
    plt.imshow(pred_matrix, cmap='gray', vmin=0, vmax=1) 
    plt.title('prediction')
    plt.axis('off')  # Optional: turn off axis labels

    plt.show()

    return

def custom_eval(label_matrix, prediction_matrix):
    diff = abs(label_matrix - prediction_matrix)
    s = np.sum(diff)
    total_pixels = label_matrix.size
    return 100 - (total_pixels / s * 100)




def a_3_pipeline():
    global path_to_ds
    path_to_ds = "datasets/dataset_9"
    global batch_size
    batch_size = 32
    """
    train_loader, val_loader, test_loader = get_dataloaders(path_to_ds, batch_size=batch_size)
    #test_data_loading(train_loader)
    #check_np_arrays()
    model = PixelClassifier()

    trained_model = train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.005)
    test_loss, test_acc = evaluate_model(trained_model, test_loader, nn.BCELoss())
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
    #TODO test on test data specified in assignment sheet
    """
    model_path = "models/run_1/ds9_ep5_lr0.005_bs32_2024-06-21_19-53-19.pth"
    test_model(model_path)

a_3_pipeline()
