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
from unet_parts import *

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
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)

    
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

    global this_run_dir_path
    this_run_dir_path = os.path.join(models_dir_name, this_run_dir_name)
    os.makedirs(this_run_dir_path)

    print("Training model...")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    training_loss_ls =  []
    val_loss_ls = []
    val_acc_ls = []
    cont_training_loss_ls = []

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
            cont_training_loss_ls.append(loss.item())
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_loss_ls.append(val_loss)
        val_acc_ls.append(val_acc)
        training_loss_ls.append(running_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    
    print("Saving plots...")
    save_plot(cont_training_loss_ls, "cont_training_loss", this_run_dir_path)
    save_plot(val_loss_ls, "val_loss", this_run_dir_path)
    save_plot(val_acc_ls, "val_acc", this_run_dir_path)
    save_plot(training_loss_ls, "training_loss", this_run_dir_path)
    print("Saving model...")
    save_model(model, num_epochs, learning_rate, this_run_dir_path)
    return model

def save_plot(data, type, this_run_dir_path):
    plt.figure(figsize=(10, 5))
    if type == "cont_training_loss":
        plt.xlabel('Training steps')
        plt.title(f'{type} Over Training Steps')
    else:
        plt.xlabel('Epochs')
        plt.title(f'{type} Over Epochs')

    plt.plot(data, label=type)
    plt.ylabel(type)
    plt.legend()
    plt.grid(True)
    path_to_plot = os.path.join(this_run_dir_path, f'{type}_plot.png')
    plt.savefig(path_to_plot)
    plt.close()

def save_model(model, num_epochs, learning_rate, this_run_dir_path):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    global model_name
    model_name = f"{model_name}_ds{path_to_ds[-1]}_ep{num_epochs}_lr{learning_rate}_bs{batch_size}_{formatted_datetime}.pth"
    path_to_model = os.path.join(this_run_dir_path, model_name)
    torch.save(model, path_to_model)

def evaluate_model(model, data_loader, criterion, save_result=False, result_path=None):
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
            all_labels.extend(labels.numpy().flatten())
            all_preds.extend(preds.numpy().flatten())
    
    loss /= len(data_loader)
    acc = accuracy_score(np.array(all_labels), np.array(all_preds))

    if save_result:
        with open(result_path, 'w') as f:
            f.write("Loss and accuracy on test data:\n")
            f.write(f'Loss: {loss}\n')
            f.write(f'Accuracy: {acc}\n')
    
    return loss, acc

def test_model(model_path):
    model = torch.load(model_path)
    path_to_label_image = "building_and_sentinel_data/Berlin_test/Berlin_test_buildings.png"
    path_to_rgb_image = "building_and_sentinel_data/Berlin_test/Berlin_test_rgb.png"
    label_im = Image.open(path_to_label_image).convert('RGB')
    label_im = np.array(label_im)

    building_mask = (label_im == [0, 0, 255]).all(axis=2) 
    label_matrix = np.zeros((label_im.shape[0], label_im.shape[1]), dtype=np.int8)
    label_matrix[building_mask] = 1

    rgb_im = Image.open(path_to_rgb_image).convert('RGB')
    rgb_im = np.array(rgb_im)
    rgb_im = rgb_im.astype(np.float32) / 255.0
    rgb_im = np.expand_dims(rgb_im, axis=0)
    rgb_im = np.transpose(rgb_im, (0, 3, 1, 2))

    inp = torch.tensor(rgb_im)

    pred = model(inp)

    pred_matrix = (pred > 0.5).float()
    pred_matrix = pred_matrix.numpy().astype(np.int8)[0, 0]

    acc = custom_acc_eval(label_matrix, pred_matrix)
    acc2 = accuracy_score(label_matrix.flatten(), pred_matrix.flatten())

    print(f"{acc*100}%")
    print(f"{acc2*100}%")

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(label_matrix, cmap='gray', vmin=0, vmax=1) 
    plt.title('labels')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(pred_matrix, cmap='gray', vmin=0, vmax=1) 
    plt.title('prediction')
    plt.axis('off')

    plt.show()

    path_to_labels = os.path.join(os.path.dirname(model_path), "labels_test_img.png")
    plt.imsave(path_to_labels, label_matrix, cmap='gray')  # Save the first image

    path_to_preds = os.path.join(os.path.dirname(model_path), "preds_test_img.png")
    plt.imsave(path_to_preds, pred_matrix, cmap='gray')  # Save the second image

def custom_acc_eval(label_matrix, prediction_matrix):
    diff = abs(label_matrix - prediction_matrix)
    wrong_preds = np.sum(diff)
    total_preds = label_matrix.size
    correct_preds = total_preds - wrong_preds
    return correct_preds / total_preds


def a_3_pipeline(train=True, model="baseline", ds="dataset_9", model_path="None"):
    global path_to_ds
    path_to_ds = f"datasets/{ds}"
    global batch_size
    batch_size = 32
    global model_name
    model_name = model

    if train:
        train_loader, val_loader, test_loader = get_dataloaders(path_to_ds, batch_size=batch_size)
        if model_name == "baseline":
            model = PixelClassifier()
        else:
            model = UNet(n_channels=3, n_classes=1, bilinear=True)
        trained_model = train_model(model, train_loader, val_loader, num_epochs=1, learning_rate=0.005)
        result_path = os.path.join(this_run_dir_path, "acc_and_loss.txt")
        test_loss, test_acc = evaluate_model(trained_model, test_loader, nn.BCELoss(), save_result=True, result_path=result_path)
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
    else:
        #model_path = "models/run_3/baseline_ds9_ep10_lr0.005_bs32_2024-06-22_21-25-37.pth"
        test_model(model_path)

a_3_pipeline(train=True, model="baseline")
