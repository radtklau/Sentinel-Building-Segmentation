import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

def prepare_tensors(city_names, patch_size=64):
    label_tensors = []
    feature_tensors = []
    for city_name in tqdm(city_names):
        path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
        path_to_rgb_image = os.path.join(path_to_city_data, f"{city_name}_rgb.png")
        path_to_label_image = os.path.join(path_to_city_data, f"{city_name}_buildings.png")
        rgb_im = Image.open(path_to_rgb_image).convert('RGB')
        rgb_im = np.array(rgb_im)
        label_im = Image.open(path_to_label_image).convert('RGB')
        label_im = np.array(label_im)

        building_mask = (label_im == [0, 0, 255]).all(axis=2) 
        label_matrix = np.zeros((rgb_im.shape[0], rgb_im.shape[1]), dtype=np.uint8)
        label_matrix[building_mask] = 1

        if patch_size > label_matrix.shape[0] or patch_size > label_matrix.shape[1]:
            print("ERROR. Patch size is bigger than image.")
            sys.exit()
        
        min_dim_len = min(label_matrix.shape)
        indices = np.arange(0, min_dim_len - patch_size, patch_size)

        num_patches = len(indices) ** 2 + 1
        label_tensor = np.zeros((num_patches, patch_size, patch_size), dtype=np.uint8)
        feature_tensor = np.zeros((num_patches, patch_size, patch_size, 3), dtype=np.uint8)

        for i in range(len(indices)):
            for index in indices:
                label_patch = label_matrix[index:index+patch_size, indices[i]:indices[i]+patch_size]
                feature_patch = rgb_im[index:index+patch_size, indices[i]:indices[i]+patch_size,:]
                label_tensor[i] = label_patch
                feature_tensor[i] = feature_patch

        label_tensor = label_tensor[1:]
        feature_tensor = feature_tensor[1:]

        print(city_name)
        label_tensor, feature_tensor = remove_cloudy_patches(label_tensor, feature_tensor)

        label_tensors.append(label_tensor)
        feature_tensors.append(feature_tensor)

    return label_tensors, feature_tensors

def build_final_tensors(label_tensors, feature_tensors):
    #stack into one tensor for labels and one for features
    #split into train, test and val dataset based on some rules (equal feature dist etc)
    #store train, test and val datasets in own dir with metadata about preparation params
    final_label_tensor = label_tensors[0]
    final_feature_tensor = feature_tensors[0]
    del label_tensors[0]
    del feature_tensors[0]

    for label_tensor, feature_tensor in zip(label_tensors, feature_tensors):
        final_label_tensor = np.concatenate((final_label_tensor, label_tensor), axis=0)
        final_feature_tensor = np.concatenate((final_feature_tensor, feature_tensor), axis=0)

    dataset_ind = 0
    while True:
        path_to_dataset = f"datasets/dataset_{dataset_ind}"
        if not os.path.exists(path_to_dataset):
            os.makedirs(path_to_dataset)
            break
        else:
            dataset_ind += 1

    feature_data_fp = os.path.join(path_to_dataset, 'features.npy')
    label_data_fp = os.path.join(path_to_dataset, 'labels.npy')
    np.save(feature_data_fp, final_feature_tensor)
    np.save(label_data_fp, final_label_tensor)

    final_label_tensor_flat = final_label_tensor.flatten()

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    features_train, features_temp, labels_train, labels_temp = train_test_split( \
        final_feature_tensor, final_label_tensor, test_size=(1 - train_size), \
        stratify=final_label_tensor_flat, random_state=42)
    
    temp_size = val_size + test_size
    val_test_split = val_size / temp_size

    labels_temp_flat = labels_temp.flatten()

    features_val, features_test, labels_val, labels_test = train_test_split( \
        features_temp, labels_temp, test_size=val_test_split, \
        stratify=labels_temp_flat, random_state=42)
    
    #TODO right now only stratification is used to balance out the labels
    #in the datasets, maybe implement further strategies to ensure balanced
    #feature distribution

    feature_data_train_fp = os.path.join(path_to_dataset, 'features_train.npy')
    label_data_train_fp = os.path.join(path_to_dataset, 'labels_train.npy')
    np.save(feature_data_train_fp, features_train)
    np.save(label_data_train_fp, labels_train)

    feature_data_test_fp = os.path.join(path_to_dataset, 'features_test.npy')
    label_data_test_fp = os.path.join(path_to_dataset, 'labels_test.npy')
    np.save(feature_data_test_fp, features_test)
    np.save(label_data_test_fp, labels_test)

    feature_data_val_fp = os.path.join(path_to_dataset, 'features_val.npy')
    label_data_val_fp = os.path.join(path_to_dataset, 'labels_val.npy')
    np.save(feature_data_val_fp, features_val)
    np.save(label_data_val_fp, labels_val)

    #TODO save metadata


def remove_cloudy_patches(label_tensor, feature_tensor):
    threshold = 0.8

    for i, patch in enumerate(feature_tensor):
        grayscale_patch = rgb2gray(patch)
        cloud_mask = grayscale_patch > threshold
        cloud_pixel_count = np.sum(cloud_mask)
        total_pixel_count = grayscale_patch.size
        cloud_percentage = (cloud_pixel_count / total_pixel_count) * 100
        if cloud_percentage > 80:
            print(f"Cloud percentage: {cloud_percentage:.2f}% in patch {i}")
            plt.imshow(patch)
            plt.show()



    return label_tensor, feature_tensor


def a_2_pipeline(city_names):
    label_tensors, feature_tensors = prepare_tensors(city_names)
    build_final_tensors(label_tensors, feature_tensors) 

global city_names
city_names = ["Berlin", "Denver", "Wien", "Helsinki", "Hamm"]
a_2_pipeline(city_names)