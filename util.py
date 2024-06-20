import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil

def plot_city_boundary_extremes(boundary_coords):
    max_longitude_point, min_longitude_point, max_latitude_point, \
    min_latitude_point = find_extreme_coords(boundary_coords)

    plt.scatter(boundary_coords[:, 0], boundary_coords[:, 1], label='All Points', color='blue')

    # Highlight the points with the highest and lowest longitude and latitude in red
    plt.scatter(max_longitude_point[0], max_longitude_point[1], color='red', label='Max Longitude')
    plt.scatter(min_longitude_point[0], min_longitude_point[1], color='red', label='Min Longitude')
    plt.scatter(max_latitude_point[0], max_latitude_point[1], color='red', label='Max Latitude')
    plt.scatter(min_latitude_point[0], min_latitude_point[1], color='red', label='Min Latitude')

    # Add labels and title for better understanding
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of Coordinates with Extremes Highlighted')
    plt.legend()

    # Display the plot
    plt.show()

def find_extreme_coords(boundary_coords):
    max_longitude_index = np.argmax(boundary_coords[:, 0])
    max_longitude_point = boundary_coords[max_longitude_index]

    # Find the point with the lowest longitude
    min_longitude_index = np.argmin(boundary_coords[:, 0])
    min_longitude_point = boundary_coords[min_longitude_index]

    # Find the point with the highest latitude
    max_latitude_index = np.argmax(boundary_coords[:, 1])
    max_latitude_point = boundary_coords[max_latitude_index]

    # Find the point with the lowest latitude
    min_latitude_index = np.argmin(boundary_coords[:, 1])
    min_latitude_point = boundary_coords[min_latitude_index]

    return max_longitude_point, min_longitude_point, max_latitude_point, min_latitude_point


def crop_im():
    path_rgb = "building_and_sentinel_data/Berlin/Berlin_rgb.png"
    path_labels = "building_and_sentinel_data/Berlin/Berlin_buildings.png"
    rgb_im = Image.open(path_rgb).convert('RGB')
    rgb_im = np.array(rgb_im)
    label_im = Image.open(path_labels).convert('RGB')
    label_im = np.array(label_im)

    label_im = label_im[1900:3000, 1500:2500]
    rgb_im = rgb_im[1900:3000, 1500:2500]

    plt.imsave("building_and_sentinel_data/Berlin/Berlin_rgb_crop.png", rgb_im)
    plt.imsave("building_and_sentinel_data/Berlin/Berlin_buildings_crop.png", label_im)

def remove_subdirs(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            # Remove the directory and its contents
            shutil.rmtree(item_path)
            print(f"Removed directory: {item_path}")
        else:
            print(f"Skipping non-directory item: {item_path}")

#remove_subdirs("datasets/")

def array_analyzer_debugging(feature_array, label_array):
    zero_count = 0
    label_im_zero_count = 0
    for i in range(feature_array.shape[0]):
        image = feature_array[i]
        l_im = label_array[i]
        red_channel_zero = np.all(image[0] == 0)
        green_channel_zero = np.all(image[1] == 0)
        blue_channel_zero = np.all(image[2] == 0)

        l_im_zero = np.all(l_im == 0)
        """
        plt.figure(figsize=(10, 5))
        
        # Plot the first image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('rgb')
        plt.axis('off')  # Optional: turn off axis labels

        # Plot the second image
        plt.subplot(1, 2, 2)
        plt.imshow(l_im, cmap='gray', vmin=0, vmax=1) 
        plt.title('labels')
        plt.axis('off')  # Optional: turn off axis labels

        plt.show()
        """
        if red_channel_zero or green_channel_zero or blue_channel_zero:
            zero_count += 1
            plt.imshow(image)
            plt.show()

        if l_im_zero:
            label_im_zero_count += 1
            plt.figure(figsize=(10, 5))
            
            # Plot the first image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('rgb')
            plt.axis('off')  # Optional: turn off axis labels

            # Plot the second image
            plt.subplot(1, 2, 2)
            plt.imshow(l_im, cmap='gray', vmin=0, vmax=1) 
            plt.title('labels')
            plt.axis('off')  # Optional: turn off axis labels

            plt.show()
        else:
            """
            plt.figure(figsize=(10, 5))
            
            # Plot the first image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('rgb')
            plt.axis('off')  # Optional: turn off axis labels

            # Plot the second image
            plt.subplot(1, 2, 2)
            plt.imshow(l_im, cmap='gray', vmin=0, vmax=1) 
            plt.title('labels')
            plt.axis('off')  # Optional: turn off axis labels

            plt.show()
            """

    print(feature_array.shape)
    print(zero_count)
    print(label_im_zero_count)

