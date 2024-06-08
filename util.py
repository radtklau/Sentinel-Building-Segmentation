import numpy as np
import matplotlib.pyplot as plt

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