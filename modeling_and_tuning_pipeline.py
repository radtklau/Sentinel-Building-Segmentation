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
import optuna
from unet_parts import *
import albumentations as A
import logging

class ImageDataset(Dataset):
    def __init__(self, path_to_ds, mode, transformation=None):
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
        #self.features = np.transpose(self.features, (0,2,3,1))

        self.transformation = transformation

    def __len__(self):
        return len(self.features)

    def __getitem__(self, ind):
        image = self.features[ind]
        label = self.labels[ind]

        if self.transformation:
            augmented = self.transformation(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        image = np.transpose(image, (2,0,1))
        return image, label

"""
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
"""

class PixelClassifier(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_stages=4, base_channels=32, dropout_prob=0.5):
        super(PixelClassifier, self).__init__()
        self.num_stages = num_stages
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        for layer_idx in range(num_stages):
            if layer_idx == 0:
                self.conv_layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1))
            else:
                self.conv_layers.append(nn.Conv2d(base_channels * (2**(layer_idx-1)),
                                                  base_channels * (2**layer_idx),
                                                  kernel_size=3, padding=1))

            # Add dropout layer after each convolutional layer (except the last one)
            if layer_idx < num_stages - 1:
                self.dropout_layers.append(nn.Dropout2d(p=dropout_prob))

        self.conv_out = nn.Conv2d(base_channels * (2**(num_stages-1)), out_channels, kernel_size=1)

    def forward(self, x):
        # Forward pass through convolutional layers
        for layer_idx in range(self.num_stages):
            x = self.conv_layers[layer_idx](x)
            x = F.relu(x)

            # Apply dropout after each convolutional layer (except the last one)
            if layer_idx < self.num_stages - 1:
                x = self.dropout_layers[layer_idx](x)

        x = self.conv_out(x)
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

    
def get_dataloaders(path_to_ds, transformation=None, batch_size=32):
    print("Creating train, test and val datasets...")
    train_dataset = ImageDataset(path_to_ds, 'train', transformation)
    val_dataset = ImageDataset(path_to_ds, 'val', transformation)
    test_dataset = ImageDataset(path_to_ds, 'test', transformation)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, learning_rate=0.001, this_run_dir_path=None, device=None, save_data=True):
    print("Training model...")
    #criterion = nn.BCELoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    running_loss_ls =  []

    val_loss_ls = []
    val_acc_ls = []
    train_loss_ls = []
    train_acc_ls = []

    cont_training_loss_ls = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            cont_training_loss_ls.append(loss.item())

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device=device)
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device=device)

        val_loss_ls.append(val_loss.cpu())
        val_acc_ls.append(val_acc)
        train_loss_ls.append(train_loss.cpu())
        train_acc_ls.append(train_acc)

        running_loss_ls.append(running_loss/len(train_loader))
        print(f'Epoch {epoch+1}/{num_epochs}, Running Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}')

    if save_data:
        print("Saving plots...")
        save_plot(train_acc_ls, "train_acc", this_run_dir_path)
        save_plot(val_acc_ls, "val_acc", this_run_dir_path)
        save_plot(train_loss_ls, "train_loss", this_run_dir_path)
        save_plot(val_loss_ls, "val_loss", this_run_dir_path)

        save_plot(cont_training_loss_ls, "cont_training_loss", this_run_dir_path, "Training step")
        save_plot(running_loss_ls, "running_loss", this_run_dir_path)

        print("Saving model...")
        save_model(model, num_epochs, learning_rate, this_run_dir_path)
    return model

def save_plot(data, type, this_run_dir_path, xlabel="Epochs"):
    plt.figure(figsize=(10, 5))
    plt.xlabel(xlabel)
    plt.title(f'{type} Over {xlabel}')

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
    #TODO change model name to include augmented data (true/false), fix bug with path_to_ds[-1]
    model_name = f"{model_name}_ds{path_to_ds[-1]}_ep{num_epochs}_lr{learning_rate}_bs{batch_size}_{formatted_datetime}.pth"
    path_to_model = os.path.join(this_run_dir_path, model_name)
    torch.save(model, path_to_model)


def evaluate_model(model, data_loader, criterion, save_result=False, this_run_dir_path=None, device=None):
    print("Evaluating model...")
    model.eval()
    loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            outputs = model(images)
            #print(type(outputs))
            loss = criterion(outputs, labels.unsqueeze(1))
            loss += loss.item()
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

    loss /= len(data_loader)
    acc = accuracy_score(np.array(all_labels), np.array(all_preds))

    if save_result:
        result_path = os.path.join(this_run_dir_path, "acc_and_loss.txt")
        with open(result_path, 'w') as f:
            f.write("Loss and accuracy on test data:\n")
            f.write(f'Loss: {loss}\n')
            f.write(f'Accuracy: {acc}\n')

    return loss, acc

"""
def test_model(model_path): #test model on test image
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
"""

def test_model(model, save_result=True, this_run_dir_path=None, device=None):
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

    inp = torch.tensor(rgb_im).to(device)
    pred = model(inp)
    #pred = pred.cpu().detach().numpy()

    #count = np.sum(pred > 0.5)
    #print(count)
    pred_matrix = (pred > 0.5).float()
    pred_matrix = pred_matrix.cpu().numpy().astype(np.int8)[0,0]
    #pred_matrix = pred_matrix.astype(np.int8)[0, 0]

    acc = accuracy_score(label_matrix.flatten(), pred_matrix.flatten())

    #print(f"{acc*100}%")

    if save_result:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(label_matrix, cmap='gray', vmin=0, vmax=1)
        plt.title('labels')
        plt.axis('off')

        path_to_plot = os.path.join(this_run_dir_path, f'labels_berlin.png')
        plt.imsave(path_to_plot, label_matrix, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_matrix, cmap='gray', vmin=0, vmax=1)
        plt.title('prediction')
        plt.axis('off')

        path_to_plot = os.path.join(this_run_dir_path, f'preds_berlin.png')
        plt.imsave(path_to_plot, pred_matrix, cmap='gray')

        plt.close()

        result_path = os.path.join(this_run_dir_path, "acc_berlin.txt")
        with open(result_path, 'w') as f:
            f.write("Accuracy on test image:\n")
            f.write(f'Accuracy: {acc}\n')



def custom_acc_eval(label_matrix, prediction_matrix):
    diff = abs(label_matrix - prediction_matrix)
    wrong_preds = np.sum(diff)
    total_preds = label_matrix.size
    correct_preds = total_preds - wrong_preds
    return correct_preds / total_preds

def objective(trial):
    print("Using GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define parameters to tune
    #hidden_dim = trial.suggest_int('hidden_dim', 16, 256, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    num_stages = trial.suggest_int('num_stages', 1, 6)
    #num_epochs = trial.suggest_int('num_epochs', 5, 20)
    dropout_prob = trial.suggest_float('dropout_prob', 0, 1)

    # Load dataset and prepare data loaders
    train_loader, val_loader, test_loader = get_dataloaders(path_to_ds, batch_size=batch_size)

    # Initialize model and optimizer
    model = PixelClassifier(in_channels=3, out_channels=1, num_stages=num_stages, base_channels=32, dropout_prob=dropout_prob)
    model.to(device)  # Move the model to the device
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss().to(device)

    trained_model = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, learning_rate=learning_rate, device=device, save_data=False)

    loss, acc = evaluate_model(trained_model, val_loader, criterion, device=device) #eval model on val data
    global logger
    logger.info(f"Trial {trial.number}: learning_rate={learning_rate}, batch_size={batch_size}, num_stages={num_stages}, dropout_prob={dropout_prob}, accuracy={acc}")
    # Report validation accuracy as the objective to minimize (negative of accuracy for maximize)
    return acc

def hyperparam_tuning():
    hyper_param_dir_name = "hyperparams"
    if not os.path.exists(hyper_param_dir_name):
        os.makedirs(hyper_param_dir_name)

    run_ind = 0
    this_run_dir_name = ""
    while True:
        path_to_run = f"hyperparams/run_{run_ind}"
        if not os.path.exists(path_to_run):
            this_run_dir_name = f"run_{run_ind}"
            break
        else:
            run_ind += 1

    this_run_dir_path = os.path.join(hyper_param_dir_name, this_run_dir_name)

    #set up logging
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    path_to_hyper_param_tuning_logs = os.path.join(this_run_dir_path, f'tuning.log')
    if not os.path.exists(path_to_hyper_param_tuning_logs):
        os.makedirs(this_run_dir_path, exist_ok=True)  # Create the directory if it doesn't exist
        open(path_to_hyper_param_tuning_logs, 'a').close()  # Create an empty file if it doesn't exist
    file_handler = logging.FileHandler(path_to_hyper_param_tuning_logs)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Get the best hyperparameters
    best_hyperparams = study.best_params
    print("Best hyperparameters:", best_hyperparams)

    path_to_best_hyper_params = os.path.join(this_run_dir_path, f'best_pixel_cl_params.txt')
    with open(path_to_best_hyper_params, 'w') as f:
        f.write("Best hyperparams for pixel classifier:\n")
        f.write(f'{best_hyperparams}\n')


def get_augmentation(aug):
    if aug == 0:
      return None
    if aug == 1:
      augmentation = A.Compose([
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.5),
      A.RandomRotate90(p=0.5),
      A.RandomBrightnessContrast(p=0.2)
      ])
    elif aug == 2:
      augmentation = A.Compose([
      A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
      A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
      A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
      ])
    elif aug == 3:
      augmentation = A.Compose([
      A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
      A.GridDistortion(p=0.5),
      A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
      ])

    return augmentation

def save_augmentation(path, transformation):
    if not transformation:
        with open(path, 'w') as f:
            f.write("No augmentation applied.\n")    
    else:
        with open(path, 'w') as f:
            f.write("Applied augmentations:\n")
            for idx, aug in enumerate(transformation, start=1):
                aug_dict = aug.to_dict()
                f.write(f"Augmentation {idx}:\n")
                f.write(f"{aug_dict['transform']}\n")


def a_3_pipeline(mode, transform=False, model="baseline", ds="dataset_25", model_path="None"):
    global path_to_ds
    global batch_size
    global model_name
    path_to_ds = f"datasets/{ds}"
    model_name = model
    batch_size = 16 #32
    learning_rate = 0.001 #0.005
    num_epochs = 20
    num_stages = 5
    dropout_prob = 0.18
    aug = 1

    print("Using GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print(device)

    if mode == "train":
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

        if transform:
            transformation = get_augmentation(aug)
        else:
            transformation = None

        augmentation_doc_path = os.path.join(this_run_dir_path, "augmentation.txt")
        save_augmentation(augmentation_doc_path, transformation)

        train_loader, val_loader, test_loader = get_dataloaders(path_to_ds, transformation, batch_size=batch_size)

        if model_name == "baseline":
            model = PixelClassifier(num_stages=num_stages, dropout_prob=dropout_prob).to(device)
            #model = PixelClassifier().to(device)
        else:
            model = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)


        criterion = nn.BCELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        trained_model = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=num_epochs, learning_rate=learning_rate, this_run_dir_path=this_run_dir_path, device=device)

        print("Testing model...")
        test_loss, test_acc = evaluate_model(trained_model, test_loader, nn.BCELoss().to(device), save_result=True, this_run_dir_path=this_run_dir_path, device=device)
        #print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

        print("Testing on Berlin...")
        test_model(trained_model, this_run_dir_path=this_run_dir_path, device=device)

    elif mode == "test":
        #model_path = "models/run_3/baseline_ds9_ep10_lr0.005_bs32_2024-06-22_21-25-37.pth"
        test_model(model_path) #TODO update passed arguments
    else: #hyperparam tuning
        hyperparam_tuning()

global model_name
a_3_pipeline(mode="train", transform=False)
#a_3_pipeline(mode="tune")