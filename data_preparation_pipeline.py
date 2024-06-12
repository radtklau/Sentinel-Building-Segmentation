import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

def prepare_tensors(city_names, patch_size=64):
    for city_name in city_names:
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
            print("ERROR. Patch is bigger than image.")
            sys.exit()
        
        label_tensor = np.zeros((patch_size, patch_size))
        feature_tensor = np.zeros((patch_size, patch_size, 3))

        min_dim_len = min(label_im.shape)
        indices = np.arange(0, min_dim_len - patch_size, patch_size)

        for i in range(len(indices)):
            for index in indices:
                label_patch = label_matrix[index:index+patch_size, indices[i]:indices[i]+patch_size]
                feature_patch = rgb_im[index:index+patch_size, indices[i]:indices[i]+patch_size,:]

                label_tensor = label_tensor.stack((label_tensor, label_patch), axis=0)
                feature_tensor = feature_tensor.stack((feature_tensor, feature_patch), axis=0)

        label_tensor = label_tensor[1:]
        feature_tensor = feature_tensor[1:]

        



def tensorize(image, labels):
    pass

def a_2_pipeline(city_names):

    prepare_tensors(city_names)
    pass

city_names = ["Berlin", "Denver", "Wien", "Helsinki", "Hamm"]
a_2_pipeline(city_names)