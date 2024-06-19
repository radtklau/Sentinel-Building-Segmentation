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
        return x

def test():
    random_data = torch.rand((1,3,64,64))
    random_data_np = random_data[0].numpy()
    random_data_np_t = np.transpose(random_data_np, (1, 2, 0))

    plt.imshow(random_data_np_t)
    plt.title('RGB Image')
    plt.axis('off')  # Optional: turn off axis labels
    plt.show()

    print(random_data)

    pxl_classifier = PixelClassifier()

    output = pxl_classifier(random_data)

    output_np = output[0].detach().numpy()
    output_np_t = np.transpose(output_np, (1, 2, 0))

    plt.imshow(output_np_t)
    plt.title('RGB Image')
    plt.axis('off')  # Optional: turn off axis labels
    plt.show()

    print(output)
    
def get_dataloaders(path_to_ds, batch_size=32):
    train_dataset = ImageDataset(path_to_ds, 'train')
    val_dataset = ImageDataset(path_to_ds, 'val')
    test_dataset = ImageDataset(path_to_ds, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}')
    
    return model

def test_data_loading(train_loader):
    for images, labels in train_loader:
        print(images.shape)
        image = images[0].permute(1,2,0).numpy()
        print(image)

        plt.imshow(image)
        plt.title('RGB Image')
        plt.axis('off')  # Optional: turn off axis labels
        plt.show()
        print(labels.shape)
        print(labels[0])

        plt.imshow(labels[0].numpy(), cmap='gray', vmin=0, vmax=1)  # Use 'gray' colormap for grayscale images
        plt.colorbar()  # Optional: add a colorbar
        plt.title("Grayscale Image")
        plt.show()

def check_np_arrays():
    path_to_ds = "datasets/dataset_28"
    features_fn = "features_train.npy"
    labels_fn = "labels_train.npy"
    features_fp = os.path.join(path_to_ds, features_fn)
    labels_fp = os.path.join(path_to_ds, labels_fn)
    features = np.load(features_fp)
    labels = np.load(labels_fp)

    for feature_im, label_map in zip(features, labels):
        feature_im = np.transpose(feature_im, (1, 2, 0))
        plt.imshow(feature_im)
        plt.title('RGB Image')
        plt.axis('off')  # Optional: turn off axis labels
        plt.show()

        plt.imshow(label_map, cmap='gray', vmin=0, vmax=1)  # Use 'gray' colormap for grayscale images
        plt.colorbar()  # Optional: add a colorbar
        plt.title("Grayscale Image")
        plt.show()




def evaluate_model(model, data_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
    
    val_loss /= len(data_loader)
    val_acc = accuracy_score(np.array(all_labels), np.array(all_preds))
    
    return val_loss, val_acc


def a_3_pipeline():
    path_to_ds = "datasets/dataset_28"
    #train_loader, val_loader, test_loader = get_dataloaders(path_to_ds, batch_size=32)
    #test_data_loading(train_loader)
    check_np_arrays() #BUG in data_preparation_pipeline.py !!! most np arrays are empty
    #model = PixelClassifier()

    #trained_model = train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)
    #val_loss, val_acc = evaluate_model(trained_model, val_loader, nn.BCELoss())
    #print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

a_3_pipeline()
