import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

def prepare_arrays(city_names, patch_size=64):
    label_arrays = []
    feature_arrays = []
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
        
        row_dim_len = label_matrix.shape[0]
        col_dim_len = label_matrix.shape[1]

        row_indices = np.arange(0, row_dim_len - patch_size, patch_size)
        col_indices = np.arange(0, col_dim_len - patch_size, patch_size)

        num_patches = len(row_indices) * len(col_indices)
        label_array = np.zeros((num_patches, patch_size, patch_size), dtype=np.uint8)
        feature_array = np.zeros((num_patches, patch_size, patch_size, 3), dtype=np.uint8)

        patch_ind = 0
        for row_ind in row_indices:
            for col_ind in col_indices:
                label_patch = label_matrix[row_ind:row_ind + patch_size, col_ind:col_ind + patch_size]
                feature_patch = rgb_im[row_ind:row_ind + patch_size, col_ind : col_ind + patch_size, :]
                label_array[patch_ind] = label_patch
                feature_array[patch_ind] = feature_patch
                patch_ind += 1

        print(city_name)
        cloud_removal_thresh = 20 #%
        label_array, feature_array = remove_patches(label_array, feature_array, cloud_removal_thresh)

        label_arrays.append(label_array)
        feature_arrays.append(feature_array)

    return label_arrays, feature_arrays

def build_final_arrays(label_arrays, feature_arrays):
    final_label_array = label_arrays[0]
    final_feature_array = feature_arrays[0]
    del label_arrays[0]
    del feature_arrays[0]

    for label_array, feature_array in zip(label_arrays, feature_arrays):
        final_label_array = np.concatenate((final_label_array, label_array), axis=0)
        final_feature_array = np.concatenate((final_feature_array, feature_array), axis=0)

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
    
    final_feature_array = np.transpose(final_feature_array, (0, 3, 1, 2)) #transpose for pytorch conv2d()
    final_feature_array = final_feature_array.astype(np.float32) / 255.0 #normalize
    final_label_array = final_label_array.astype(np.float32)

    np.save(feature_data_fp, final_feature_array)
    np.save(label_data_fp, final_label_array)

    #final_label_array_flat = final_label_array.flatten()

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    features_train, features_temp, labels_train, labels_temp = train_test_split( \
        final_feature_array, final_label_array, test_size=(1 - train_size), random_state=42)

    temp_size = val_size + test_size
    val_test_split = val_size / temp_size

    #labels_temp_flat = labels_temp.flatten()

    features_val, features_test, labels_val, labels_test = train_test_split( \
        features_temp, labels_temp, test_size=val_test_split, random_state=42)
    
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


def remove_patches(label_array, feature_array, cloud_removal_thresh):
    cloud_thresh = 0.95
    total_num_of_patches = feature_array.shape[0]

    num_patches_removed_cloud = 0
    num_patches_removed_black = 0
    num_patches_removed_no_label = 0
    to_delete = []
    for i, patch in enumerate(feature_array):
        label_patch = label_array[i]
        label_patch_zero = np.all(label_patch == 0)

        red_zero = np.all(patch[0] == 0)
        green_zero = np.all(patch[1] == 0)
        blue_zero = np.all(patch[2] == 0)

        grayscale_patch = rgb2gray(patch)
        cloud_mask = grayscale_patch > cloud_thresh
        cloud_pixel_count = np.sum(cloud_mask)
        total_pixel_count = grayscale_patch.size
        cloud_percentage = (cloud_pixel_count / total_pixel_count) * 100

        if cloud_percentage > cloud_removal_thresh: #cloud removal
            to_delete.append(i)
            #print(f"Cloud percentage: {cloud_percentage:.2f}% in patch {i}")
            #plt.imshow(patch)
            #plt.show()
            num_patches_removed_cloud += 1
        elif red_zero or green_zero or blue_zero: #black rgb image removal
            #print(f"Red channel is zero: {red_zero}, blue channel is zero: {blue_zero}, green channel is zero: {green_zero}")
            to_delete.append(i)
            num_patches_removed_black += 1
        elif label_patch_zero: #no label removal
            #print(f"Removing patch {i} with no labels.")
            to_delete.append(i)
            num_patches_removed_no_label += 1      
        
    feature_array = np.delete(feature_array, to_delete, axis=0)
    label_array = np.delete(label_array, to_delete, axis=0)

    print(f"Removed {num_patches_removed_cloud} cloudy patches, {num_patches_removed_black} black \
          patches and {num_patches_removed_no_label} patches with no label.\n(total number of patches: {total_num_of_patches},\npatches remaining: {feature_array.shape[0]})\n")
    
    return label_array, feature_array


def a_2_pipeline(city_names):
    label_arrays, feature_arrays = prepare_arrays(city_names)
    build_final_arrays(label_arrays, feature_arrays) 

global city_names
city_names = ["Berlin", "Denver", "Wien", "Helsinki", "Hamm"]
a_2_pipeline(city_names)