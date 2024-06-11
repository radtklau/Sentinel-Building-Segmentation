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
from pyproj import Proj
import math
from shapely.geometry import Polygon
import rasterio.features
from geopy.distance import geodesic
import geopandas as gpd

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
        #pickle.dump(building_geometry, f)
        pickle.dump(buildings_in_city_boundaries, f)

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


def find_extreme_points():
    boundary_coords = np.array(city_boundary["geometry"].exterior.coords)

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

    return min_longitude_point, min_latitude_point, max_longitude_point, max_latitude_point
    

def get_sentinel_image_data(temporal_extent, bands, city_name):
    print("Connecting to backend...")
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu/openeo/1.2").authenticate_oidc()

    print("Extracting city coordinates...")
    global city_boundary
    min_longitude_point, min_latitude_point, max_longitude_point, max_latitude_point = find_extreme_points()

    """
    min_longitude_point = city_boundary["geometry"].bounds[0]
    min_latitude_point = city_boundary["geometry"].bounds[1]
    max_longitude_point = city_boundary["geometry"].bounds[2]
    max_latitude_point = city_boundary["geometry"].bounds[3]
    """
    #global spatial_extent
    spatial_extent = {"west":min_longitude_point[0], "south":min_latitude_point[1], \
                      "east":max_longitude_point[0], "north":max_latitude_point[1]} 
    
    print(spatial_extent)

    datacube = connection.load_collection("SENTINEL2_L2A", \
            spatial_extent=spatial_extent, \
            temporal_extent=temporal_extent, \
            bands=bands
            ) #BUG doesnt precisely deliver image in spatial_extent corners! compare Flensburg rgb top right corner with geo coords of corner on google maps!
    
    #job = datacube.export(filename=f"{city_name}_sentinel.tiff", format="GTiff")
    #job.start_and_wait().download_results()

    
    building_and_sentinel_data_dir_name = "building_and_sentinel_data"
    if not os.path.exists(building_and_sentinel_data_dir_name):
        os.makedirs(building_and_sentinel_data_dir_name)

    building_and_sentinel_city_data_dir_name = \
        f"{building_and_sentinel_data_dir_name}/{city_name}"
    if not os.path.exists(building_and_sentinel_city_data_dir_name):
        os.makedirs(building_and_sentinel_city_data_dir_name)
    
    print("Downloading image data...")
    result = datacube.download(os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}.tiff"))

    """
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

    print("Writing image data to disk...")
    
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
    """
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
        utm_crs = f'326{utm_zone}'
    else:
        utm_crs = f'327{utm_zone}'

    geographic_crs = 'EPSG:4326'
    #utm_crs = 'EPSG:25832'

    # Create a transformer object for the conversion
    transformer = Transformer.from_crs(geographic_crs, utm_crs, always_xy=True)

    if polygon:
        geo_df = gpd.GeoDataFrame(geometry=building_geometry, crs=4326)
        utm_geo_df = geo_df.to_crs(utm_crs)
        """
        utm_coords = []
        for polygon in building_geometry.values.data:
            building_coords = np.array(polygon.exterior.coords)
            polygon_utm_coords = []
            for coords in building_coords:
                utm_x, utm_y = transformer.transform(coords[0], coords[1])
                utm_coord = [utm_x, utm_y]
                polygon_utm_coords.append(utm_coord)

            utm_coords.append(polygon_utm_coords)
        """
        return utm_geo_df["geometry"].values.data
    
    else:
        utm_min_x, _ = transformer.transform(im_corners[0], im_corners[1])
        _, utm_max_y = transformer.transform(im_corners[0], im_corners[3])
        #utm_min_x, utm_max_y = transformer.transform(im_corners[0], im_corners[3])
        #utm_min_x *= 0.9996
        #utm_max_y *= 0.9996
        return [utm_min_x, utm_max_y]
    
def build_stacked_im(city_name, label_im):
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_rgb_image = os.path.join(path_to_city_data, f"{city_name}_rgb.png")
    rgb_im = Image.open(path_to_rgb_image).convert('RGB')  # Convert to RGB to remove the alpha channel
    rgb_im = np.array(rgb_im)

    # Create a mask for the white pixels in the BW image
    mask = label_im == 1

    # Prepare an output image
    stacked_im = np.copy(rgb_im)

    # Set the white pixels in the BW image to white in the RGB image
    stacked_im[mask] = [0, 0, 255]

    path_to_stacked_image = os.path.join(path_to_city_data, f"{city_name}_stacked_new2.png")
    plt.imsave(path_to_stacked_image, stacked_im)

def extract_coordinates(geometry):
    if geometry.geom_type == 'Polygon':
        exterior_coords = list(geometry.exterior.coords)
        return exterior_coords
    elif geometry.geom_type == 'MultiPolygon':
        # Handle MultiPolygon by iterating over each polygon
        all_coords = []
        for polygon in geometry:
            all_coords.extend(list(polygon.exterior.coords))
        return all_coords
    else:
        return []
    
def coords_to_rowcol(coords, transformer):
    row_col_list = []
    for coord in coords:
        x, y = coord
        row, col = transformer.rowcol(x, y)
        row_col_list.append((row, col))
    return row_col_list


def label_gen(city_name):
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_building_data = os.path.join(path_to_city_data, f"{city_name}_buildings.pkl")
    path_to_tiff_im = os.path.join(path_to_city_data, f"{city_name}.tiff")
    tiff_img = rasterio.open(path_to_tiff_im)

    print(tiff_img.crs)
    print(tiff_img.transform)

    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_building_data = os.path.join(path_to_city_data, f"{city_name}_buildings.pkl")
    buildings_in_city_boundaries = None
    try:
        with open(path_to_building_data, 'rb') as f:
            buildings_in_city_boundaries = pickle.load(f)
    except Exception as e:
        print(f"An error occurred: {e}")

    utm_buildings = buildings_in_city_boundaries.to_crs(tiff_img.crs)
    transformer = rasterio.transform.AffineTransformer(tiff_img.transform)

    utm_buildings['coords'] = utm_buildings['geometry'].apply(extract_coordinates)

    utm_buildings['row_col'] = utm_buildings['coords'].apply(lambda coords: coords_to_rowcol(coords, transformer))

    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_rgb_image = os.path.join(path_to_city_data, f"{city_name}_rgb.png")
    rgb_im = Image.open(path_to_rgb_image).convert('RGB')  # Convert to RGB to remove the alpha channel
    rgb_im = np.array(rgb_im)

    #fig, ax = plt.subplots(figsize=(10, 10))

    x = list(utm_buildings['row_col'])
    # Plot the GeoDataFrame with customizations
    #utm_buildings['row_col'].plot(ax=ax)
    polygons = [Polygon(coords) for coords in utm_buildings['row_col']]
    pixel_polygons = []
    for building in x:
        pixel_polygon = []
        for coord in building:
            x = coord[0]
            y = coord[1]
            pixel_coord = (y,x)
            #print(pixel_coord)
            pixel_polygon.append(pixel_coord)
        pixel_polygons.append(Polygon(pixel_polygon))

    label_im = rasterio.features.rasterize(pixel_polygons, out_shape=rgb_im.shape[:2], all_touched=True)
    #label_im = np.rot90(label_im, k=1)
    #label_im = np.fliplr(label_im)
    build_stacked_im(city_name, label_im)
    """
    label_im = np.zeros(rgb_im.shape[:2])
     
    for building in x:
        for coord in building:
            label_im[coord] = 1

    build_stacked_im(city_name, label_im)
    #build_stacked_im(city_name, label_im)
    #path_to_stacked_image = os.path.join(path_to_city_data, f"{city_name}_stacked_new3.png")
    #plt.imsave(path_to_stacked_image, rgb_im)
    """
    sys.exit()

    

    #utm_polygon_coords = geo_coord_reprojection(city_name)



    global city_boundary
    min_longitude_point, min_latitude_point, max_longitude_point, max_latitude_point = find_extreme_points()

    im_corners = [min_longitude_point[0], min_latitude_point[1], max_longitude_point[0], max_latitude_point[1]]
    utm_im_corners = geo_coord_reprojection(city_name, polygon=False, im_corners=im_corners)

    pixel_resolution = 10

    pixel_polygons = []
    for utm_polygon_coord in utm_polygon_coords:
        pixel_polygon = []
        for utm_coord in np.array(utm_polygon_coord.exterior.coords):
            utm_x_coord = utm_coord[0]
            utm_y_coord = utm_coord[1]
            x_dist_im_corner = utm_x_coord - utm_im_corners[0]
            y_dist_im_corner = abs(utm_y_coord - utm_im_corners[1])

            x_pixel_coord = math.floor(x_dist_im_corner / pixel_resolution)
            y_pixel_coord = math.floor(y_dist_im_corner / pixel_resolution)

            pixel_coord = (x_pixel_coord, y_pixel_coord)
            pixel_polygon.append(pixel_coord)

        pixel_polygons.append(Polygon(pixel_polygon))

    #REMOVE#
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_rgb_image = os.path.join(path_to_city_data, f"{city_name}_r.png")
    r_im = Image.open(path_to_rgb_image)
    r_im = np.array(r_im)      
    #global im_shape
    im_shape = r_im.shape[:2]  
    #REMOVE#

    label_im = rasterio.features.rasterize(pixel_polygons, out_shape=im_shape, all_touched=True)

    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    path_to_building_png = os.path.join(path_to_city_data, f"{city_name}_buildings.png")
    plt.imsave(path_to_building_png, label_im, cmap='gray')
    print("Building stacked image...")
    build_stacked_im(city_name, label_im)

def a_1_pipeline(city_name):
    global city_boundary
    global im_shape

    get_osm_building_data(city_name)
    #plot_building_data(city_name)

    temporal_extent=["2024-05-13", "2024-05-14"]
    bands=["B04", "B03", "B02", "B08"]

    get_sentinel_image_data(temporal_extent, bands, city_name)
    label_gen(city_name)

city_name = "Hamm"
a_1_pipeline(city_name)
#plot_data("Hamm", "buildings")