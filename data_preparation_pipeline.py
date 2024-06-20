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

        plt.imshow(rgb_im)  
        plt.title("rgb_im")
        plt.show()

        plt.imshow(label_im, cmap='gray', vmin=0, vmax=1)
        plt.title("label_im")
        plt.show()

        building_mask = (label_im == [0, 0, 255]).all(axis=2) 
        label_matrix = np.zeros((rgb_im.shape[0], rgb_im.shape[1]), dtype=np.uint8)
        label_matrix[building_mask] = 1

        if patch_size > label_matrix.shape[0] or patch_size > label_matrix.shape[1]:
            print("ERROR. Patch size is bigger than image.")
            sys.exit()
        
        min_dim_len = min(label_matrix.shape)
        indices = np.arange(0, min_dim_len - patch_size, patch_size)

        num_patches = len(indices) ** 2 + 1
        label_array = np.zeros((num_patches, patch_size, patch_size), dtype=np.uint8)
        feature_array = np.zeros((num_patches, patch_size, patch_size, 3), dtype=np.uint8)

        patch_ind = 0
        for row_ind in indices:
            for col_ind in indices:
                label_patch = label_matrix[row_ind:row_ind + patch_size, col_ind:col_ind + patch_size]
                feature_patch = rgb_im[row_ind:row_ind + patch_size, col_ind : col_ind + patch_size, :]
                label_array[patch_ind] = label_patch
                feature_array[patch_ind] = feature_patch
                patch_ind += 1

        label_array = label_array[1:]
        feature_array = feature_array[1:]
        array_analyzer_debugging(feature_array, label_array)
        print(city_name)
        removal_thresh = 10 #%
        label_array, feature_array = remove_cloudy_patches(label_array, feature_array, removal_thresh)

        

        label_arrays.append(label_array)
        feature_arrays.append(feature_array)

    return label_arrays, feature_arrays

def array_analyzer_debugging(feature_array, label_array):
    zero_count = 0
    for i in range(feature_array.shape[0]):
        image = feature_array[i]
        red_channel_zero = np.all(image[0] == 0)
        green_channel_zero = np.all(image[1] == 0)
        blue_channel_zero = np.all(image[2] == 0)
        if red_channel_zero or green_channel_zero or blue_channel_zero:
            zero_count += 1
            #plt.imshow(image)  # Use 'gray' colormap for grayscale images
            #plt.show()

    print(feature_array.shape)
    print(zero_count)

def build_final_arrays(label_arrays, feature_arrays):
    #stack into one array for labels and one for features
    #split into train, test and val dataset based on some rules (equal feature dist etc)
    #store train, test and val datasets in own dir with metadata about preparation params
    final_label_array = label_arrays[0]
    final_feature_array = feature_arrays[0]
    del label_arrays[0]
    del feature_arrays[0]

    for label_array, feature_array in zip(label_arrays, feature_arrays):
        final_label_array = np.concatenate((final_label_array, label_array), axis=0)
        final_feature_array = np.concatenate((final_feature_array, feature_array), axis=0)


    array_analyzer_debugging(final_feature_array, final_label_array)

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


def remove_cloudy_patches(label_array, feature_array, removal_thresh):
    cloud_thresh = 0.8

    for i, patch in enumerate(feature_array):
        grayscale_patch = rgb2gray(patch)
        cloud_mask = grayscale_patch > cloud_thresh
        cloud_pixel_count = np.sum(cloud_mask)
        total_pixel_count = grayscale_patch.size
        cloud_percentage = (cloud_pixel_count / total_pixel_count) * 100
        if cloud_percentage > removal_thresh:
            feature_array = np.delete(feature_array, i, axis=0)
            label_array = np.delete(label_array, i, axis=0)
            print(f"Cloud percentage: {cloud_percentage:.2f}% in patch {i}")
            #plt.imshow(patch)
            #plt.show()

    return label_array, feature_array


def a_2_pipeline(city_names):
    label_arrays, feature_arrays = prepare_arrays(city_names)
    build_final_arrays(label_arrays, feature_arrays) 

global city_names
city_names = ["Berlin", "Denver", "Wien", "Helsinki", "Hamm"]
a_2_pipeline(city_names)