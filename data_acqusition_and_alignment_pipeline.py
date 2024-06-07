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
import openeo
import imageio

def get_osm_building_data(city_name):
    if city_name not in sources.cities.available:
        print(f"{city_name} not available.")
        sys.exit()

    osm_data_dir_name = "city_osm_data"
    if not os.path.exists(osm_data_dir_name):
        os.makedirs(osm_data_dir_name)
    
    try:
        print(f"Loading {city_name} osm data...")
        fp = get_data(city_name, directory=osm_data_dir_name)
    except Exception as e:
        print(f"An error occurred: {e}")

    global city_boundary
    city_boundary = get_city_boundary(fp, city_name)

    #print(city_boundary["name"])
    osm_buildings = OSM(fp, bounding_box=city_boundary["geometry"])
    print("Extracting buildings in city boundary...")
    buildings_in_city_boundaries = osm_buildings.get_buildings()

    building_geometry = buildings_in_city_boundaries["geometry"]

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

def get_city_boundary(fp, city_name):
    osm = OSM(fp)
    print("Extracting city boundaries...")
    boundaries_city = osm.get_boundaries(name=city_name)

    #boundaries_city contains all bounds that contain the city name
    #assume that bound with city tag is actual city bound, if no city
    #tag exists, assume that bound wiht biggest area is the actual city bound
    ind_city_status = -1
    city_tag_encountered = 0
    ind_biggest_area = -1
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
            pass

        area_in_bound = row["geometry"].area
        if area_in_bound > max_area:
            max_area = area_in_bound
            ind_biggest_area = index

    if city_tag_encountered == 1:
        city_boundary = boundaries_city.iloc[ind_city_status]
    else:
        if ind_biggest_area == -1:
            print("Error: No city bound found")
            sys.exit()
        city_boundary = boundaries_city.iloc[ind_biggest_area]

    return city_boundary

def plot_building_data(city_name):
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

def plot_data(city_name, type): #plot relevant data
    if type not in ["rgb", "irb", "buildings", "rgb_buildings"]:
        pass
    pass

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

def get_sentinel_image_data(temporal_extent, bands, city_name):
    print("Connecting to backend...")
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu/openeo/1.2").authenticate_oidc()

    print("Extracting city coordinates...")
    global city_boundary
    min_longitude_point = city_boundary["geometry"].bounds[0]
    min_latitude_point = city_boundary["geometry"].bounds[1]
    max_longitude_point = city_boundary["geometry"].bounds[2]
    max_latitude_point = city_boundary["geometry"].bounds[3]

    spatial_extent = {"west":min_longitude_point, "south":min_latitude_point, \
                      "east":max_longitude_point, "north":max_latitude_point} 
    
    datacube = connection.load_collection("SENTINEL2_L2A", \
            spatial_extent=spatial_extent, \
            temporal_extent=temporal_extent, \
            bands=bands
            )
    
    print("Downloading image data...")
    result = datacube.download()

    print("Processing image data...")
    image_array = imageio.imread(result)
    clipped_image = np.clip(image_array, 0, 2500)

    clipped_image = clipped_image.astype(np.float64)
    clipped_image /= 2500
    clipped_image *= 255

    rgb_im = np.stack([clipped_image[0], clipped_image[1], clipped_image[2]], axis=-1)
    irb_im = np.stack([clipped_image[3], clipped_image[0], clipped_image[1]], axis=-1)
    r_im = clipped_image[0]
    g_im = clipped_image[1]
    b_im = clipped_image[2]
    vnir_im = clipped_image[3]

    rgb_im = rgb_im.astype(np.uint8)
    irb_im = irb_im.astype(np.uint8)
    r_im = r_im.astype(np.uint8)
    g_im = g_im.astype(np.uint8)
    b_im = b_im.astype(np.uint8)
    vnir_im = vnir_im.astype(np.uint8)

    print("Writing to disk...")
    building_and_sentinel_data_dir_name = "building_and_sentinel_data"
    if not os.path.exists(building_and_sentinel_data_dir_name):
        os.makedirs(building_and_sentinel_data_dir_name)

    building_and_sentinel_city_data_dir_name = \
        f"{building_and_sentinel_data_dir_name}/{city_name}"
    if not os.path.exists(building_and_sentinel_city_data_dir_name):
        os.makedirs(building_and_sentinel_city_data_dir_name)
    
    satellite_data_rgb_fp = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_rgb.png")
    satellite_data_r_fp = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_r.png")
    satellite_data_g_fp = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_g.png")
    satellite_data_b_fp = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_b.png")
    satellite_data_irb_fp = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_irb.png")
    satellite_data_vnir_fp = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_vnir.png")

    plt.imsave(satellite_data_rgb_fp, rgb_im)
    plt.imsave(satellite_data_r_fp, r_im, cmap='gray')
    plt.imsave(satellite_data_g_fp, g_im, cmap='gray')
    plt.imsave(satellite_data_b_fp, b_im, cmap='gray')
    plt.imsave(satellite_data_irb_fp, irb_im)
    plt.imsave(satellite_data_vnir_fp, vnir_im, cmap='gray')


def a_1_pipeline(city_name):
    global city_boundary

    get_osm_building_data(city_name)
    #plot_building_data(city_name)

    #spatial_extent={"west": 13.10, "south": 52.35, "east": 13.66, "north": 52.64} (berlin)
    temporal_extent=["2024-05-13", "2024-05-14"]
    bands=["B04", "B03", "B02", "B08"]

    get_sentinel_image_data(temporal_extent, bands, city_name)

city_name = "Hamm"
a_1_pipeline(city_name)

#TODO download of infrared band,
#plotting pipelines for the report of single bands (Figure 2a), buildings (Figure 1a), RGB (Figure
#1b), IRB (Infrared, Red, and Blue) (Figure 2c), and overlapping buildings (Figure 2b)



