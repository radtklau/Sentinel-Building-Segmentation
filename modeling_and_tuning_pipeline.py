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
    
def get_dataloaders(train_images, train_labels, val_images, val_labels, batch_size=32):
    train_dataset = PixelClassifier(train_images, train_labels)
    val_dataset = PixelClassifier(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

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

"""
def a_3_pipeline():
    train_images = np.random.rand(100, 3, 64, 64)  # Example data
    train_labels = np.random.randint(0, 2, (100, 64, 64))  # Example labels
    val_images = np.random.rand(20, 3, 64, 64)  # Example data
    val_labels = np.random.randint(0, 2, (20, 64, 64))  # Example labels

    train_loader, val_loader = get_dataloaders(train_images, train_labels, val_images, val_labels, batch_size=8)

    model = PixelClassifier()

    trained_model = train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001)

    val_loss, val_acc = evaluate_model(trained_model, val_loader, nn.BCELoss())
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

a_3_pipeline()
"""