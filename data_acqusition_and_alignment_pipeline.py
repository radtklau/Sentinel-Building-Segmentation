from pyrosm.data import sources
from pyrosm import get_data
from tqdm import tqdm
from pyrosm import OSM
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import ast
import sys

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
    boundaries_city = osm.get_boundaries(name=city_name)

    #boundaries_city contains all bounds that contain the city name
    #assume that bound with city tag is actual city bound, if no city
    #tag exists, assume that bound wiht biggest area is the actual city bound
    ind_city_status = -1
    city_tag_encountered = 0
    ind_biggest_area = 0
    max_area = 0
    for index,row in boundaries_city.iterrows():
        try:
            tags = row["tags"]
            if isinstance(tags, str):
                tags = ast.literal_eval(tags)
            status = tags["de:place"]
            if status == "city":
                city_tag_encountered += 1
                ind_city_status = index
            else:
                continue #if status != city, we dont want to consider the size of the area
        except: #no place tag, consider size of area
            continue

        area_in_bound = row["geometry"].area
        if area_in_bound > max_area:
            max_area = area_in_bound
            ind_biggest_area = index

    if city_tag_encountered == 1:
        city_bound = boundaries_city.iloc[ind_city_status]
    else:
        if ind_biggest_area == 0:
            print("Error: No city bound found")
            sys.exit()
        city_bound = boundaries_city.iloc[ind_biggest_area]

    #print(city_bound["name"])
    osm_buildings = OSM(fp, bounding_box=city_bound["geometry"])
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

def plot_city_building_data(city_name):
    fp = f"building_and_sentinel_data/{city_name}/{city_name}_buildings.pkl"
    print("Plotting building data...")

    building_geometry = None
    try:
        with open(fp, 'rb') as f:
            building_geometry = pickle.load(f)
    except Exception as e:
        print(f"An error occurred: {e}")

    fig, ax = plt.subplots()
    building_geometry.plot(ax=ax)
    #ax.set_aspect('equal')

    plt.show()

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

def get_sentinel_city_data():
    pass

city_name = "Hamm"
get_osm_city_building_data(city_name)
plot_city_building_data(city_name)




