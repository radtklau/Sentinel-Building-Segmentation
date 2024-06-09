from pyrosm.data import sources
from pyrosm import get_data
from tqdm import tqdm
from pyrosm import OSM
import geopandas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pickle
import ast
import sys
import openeo
import imageio
from PIL import Image
from pyproj import Transformer
import math
from shapely.geometry import Polygon
import rasterio.features

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
    buildings_in_city_boundaries = osm_buildings.get_buildings() #EPGS:4326

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


def plot_data(city_name, type): #plot relevant data
    if type not in ["rgb", "irb", "buildings", "rgb_buildings", "r", "g", "b"]:
        print("Not a valid type.")
        return
    
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)

    if type == "buildings":
        path_to_building_data = os.path.join(path_to_city_data, f"{city_name}_buildings.pkl")
        building_geometry = None
        try:
            with open(path_to_building_data, 'rb') as f:
                building_geometry = pickle.load(f)
        except Exception as e:
            print(f"An error occurred: {e}")

        _, ax = plt.subplots()
        building_geometry.plot(ax=ax)
        plt.show()
    else:
        path_to_image = os.path.join(path_to_city_data, f"{city_name}_{type}.png")
        image = mpimg.imread(path_to_image)
        plt.imshow(image)
        plt.axis('off')
        plt.show()


def buildings_pkl_to_png(city_name):
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_building_pkl = os.path.join(path_to_city_data, f"{city_name}_buildings.pkl")
    path_to_building_png = os.path.join(path_to_city_data, f"{city_name}_buildings.png")
    building_geometry = None
    try:
        with open(path_to_building_pkl, 'rb') as f:
            building_geometry = pickle.load(f)
    except Exception as e:
        print(f"An error occurred: {e}")

    fig, ax = plt.subplots()

    # Set the figure background to transparent
    #fig.patch.set_alpha(0.0)

    # Plot the building geometry
    if building_geometry is not None:
        building_geometry.plot(ax=ax)

    # Set the axes background to transparent
    #ax.patch.set_alpha(0.0)

    #ax.axis('off')

    # Save the plot as PNG with transparent background
    plt.savefig(path_to_building_png, transparent=True)
    plt.close(fig)

def build_stacked_im(city_name): #TODO
    buildings_pkl_to_png(city_name)
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_rgb_image = os.path.join(path_to_city_data, f"{city_name}_rgb.png")
    rgb_im = Image.open(path_to_rgb_image)

    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_building_data = os.path.join(path_to_city_data, f"{city_name}_buildings.png")
    building_im = Image.open(path_to_building_data)

    if rgb_im.size != building_im.size:
        print(rgb_im.size)
        print(building_im.size)
        print("ohhhhhhhhhhhhhhh")
        building_im = building_im.resize(rgb_im.size)

    stacked_im = Image.alpha_composite(rgb_im.convert('RGBA'), building_im)

    path_to_stacked_im = os.path.join(path_to_city_data, f"{city_name}_rgb_buildings.png")
    stacked_im.save(path_to_stacked_im)

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

    global im_shape
    im_shape = r_im.shape
    print(im_shape)

    print("Building stacked image...")
    #build_stacked_im(city_name)
    buildings_rgb_stacked_im = 0 #TODO

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

def geo_coord_reprojection(city_name, polygon=True, im_corners=None):
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_building_data = os.path.join(path_to_city_data, f"{city_name}_buildings.pkl")
    building_geometry = None
    try:
        with open(path_to_building_data, 'rb') as f:
            building_geometry = pickle.load(f)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    max_lon = building_geometry.total_bounds[2]
    min_lon = building_geometry.total_bounds[0]
    half_lon = (max_lon - min_lon) / 2 + min_lon

    max_lat = building_geometry.total_bounds[3]
    min_lat = building_geometry.total_bounds[1]
    half_lat = (max_lat - min_lat) / 2 + min_lat
    
    utm_zone = math.floor((half_lon + 180) / 6) + 1

    if half_lat > 0:
        utm_crs = f'EPSG:326{utm_zone}'
    else:
        utm_crs = f'EPSG:327{utm_zone}'

    geographic_crs = 'EPSG:4326'
    #utm_crs = 'EPSG:1078'

    # Create a transformer object for the conversion
    transformer = Transformer.from_crs(geographic_crs, utm_crs, always_xy=True)

    if polygon:
        utm_coords = []

        for polygon in building_geometry.values.data:
            building_coords = np.array(polygon.exterior.coords)
            polygon_utm_coords = []
            for coords in building_coords:
                utm_x, utm_y = transformer.transform(coords[0], coords[1])
                utm_coord = [utm_x, utm_y]
                polygon_utm_coords.append(utm_coord)

            utm_coords.append(polygon_utm_coords)

        return utm_coords
    
    else:
        utm_min_x, utm_min_y = transformer.transform(im_corners[0], im_corners[1])
        utm_max_x, utm_max_y = transformer.transform(im_corners[2], im_corners[3])
        return [utm_min_x, utm_min_y, utm_max_x, utm_max_y]

def label_gen(city_name):
    #label_im = np.zeros(im_shape)
    utm_polygons = geo_coord_reprojection(city_name)

    global city_boundary
    min_longitude_point = city_boundary["geometry"].bounds[0]
    min_latitude_point = city_boundary["geometry"].bounds[1]
    max_longitude_point = city_boundary["geometry"].bounds[2]
    max_latitude_point = city_boundary["geometry"].bounds[3]

    im_corners = [min_longitude_point, min_latitude_point, max_longitude_point, max_latitude_point]
    utm_im_corners = geo_coord_reprojection(city_name, polygon=False, im_corners=im_corners)
    print("x")
    pixel_resolution = 10

    pixel_polygons = []
    for utm_polygon in utm_polygons:
        pixel_polygon = []
        for utm_coord in utm_polygon:
            x_coord = utm_coord[0]
            y_coord = utm_coord[1]
            x_dist_im_corner = x_coord - utm_im_corners[0]
            y_dist_im_corner = y_coord - utm_im_corners[1]

            x_pixel_coord = math.floor(x_dist_im_corner / pixel_resolution)
            y_pixel_coord = math.floor(y_dist_im_corner / pixel_resolution)
            #label_im[y_pixel_coord, x_pixel_coord] = 1

            pixel_coord = (y_pixel_coord, x_pixel_coord)
            pixel_polygon.append(pixel_coord)

        pixel_polygons.append(Polygon(pixel_polygon))
    
    label_im = rasterio.features.rasterize(pixel_polygons, out_shape=im_shape)

    plt.imshow(label_im, cmap='gray', interpolation='nearest')
    #plt.axis('off')
    plt.imsave("test.png", label_im, cmap='gray')
    plt.show()




def a_1_pipeline(city_name):
    global city_boundary
    global im_shape

    get_osm_building_data(city_name)
    #plot_building_data(city_name)

    #spatial_extent={"west": 13.10, "south": 52.35, "east": 13.66, "north": 52.64} (berlin)
    temporal_extent=["2024-05-13", "2024-05-14"]
    bands=["B04", "B03", "B02", "B08"]

    get_sentinel_image_data(temporal_extent, bands, city_name)
    label_gen(city_name)

city_name = "Hamm"
a_1_pipeline(city_name)
#build_stacked_im("Hamm")
#plot_data("Hamm", "buildings")



#TODO take OSM polygons and project into sentinel data projection


