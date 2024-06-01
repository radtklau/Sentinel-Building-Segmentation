from pyrosm.data import sources
from pyrosm import get_data
from tqdm import tqdm
from pyrosm import OSM
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def get_osm_city_building_data(city_name):
    osm_data_dir_name = "city_osm_data"
    if not os.path.exists(osm_data_dir_name):
        os.makedirs(osm_data_dir_name)
    
    try:
        print(f"Loading {city_name} osm data...")
        fp = get_data(city_name, directory=osm_data_dir_name)
    except Exception as e:
        print(f"An error occurred: {e}")

    osm = OSM(fp)
    print("Extracting city boundaries...")
    boundaries = osm.get_boundaries()
    boundaries_city = boundaries[boundaries['name']==city_name]
    #boundary_coords = np.array(boundaries_city["geometry"].iloc[0].exterior.coords)

    osm_buildings = OSM(fp, bounding_box=boundaries_city['geometry'].values[0])
    print("Extracting buildings in city boundary...")
    buildings_in_city_bounds = osm_buildings.get_buildings()

    building_geometry = buildings_in_city_bounds["geometry"]

    print("Storing building data...")
    building_and_sentinel_data_dir_name = "building_and_sentinel_data"
    if not os.path.exists(building_and_sentinel_data_dir_name):
        os.makedirs(building_and_sentinel_data_dir_name)

    building_and_sentinel_city_data_dir_name = \
        f"{building_and_sentinel_data_dir_name}/{city_name}"
    if not os.path.exists(building_and_sentinel_city_data_dir_name):
        os.makedirs(building_and_sentinel_city_data_dir_name)
    
    building_data_fp = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_buildings.pkl")

    with open(building_data_fp, 'wb') as f:
        pickle.dump(building_geometry, f)

get_osm_city_building_data("Jena")

def plot_city_boundary_extremes(boundary_coords):

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


