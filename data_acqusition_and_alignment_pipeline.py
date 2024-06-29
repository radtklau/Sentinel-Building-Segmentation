from pyrosm.data import sources
from pyrosm import get_data
from pyrosm import OSM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import pickle
import ast
import sys
import openeo
from PIL import Image
from shapely.geometry import Polygon
import rasterio.features


def get_osm_building_data(city_name):
    if city_name not in sources.cities.available:
        print(f"{city_name} not available.")
        sys.exit()

    building_and_sentinel_data_dir_name = "building_and_sentinel_data"
    if not os.path.exists(building_and_sentinel_data_dir_name):
        os.makedirs(building_and_sentinel_data_dir_name)

    building_and_sentinel_city_data_dir_name = \
        f"{building_and_sentinel_data_dir_name}/{city_name}"
    if not os.path.exists(building_and_sentinel_city_data_dir_name):
        os.makedirs(building_and_sentinel_city_data_dir_name)

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
    osm_buildings = OSM(fp, bounding_box=city_boundary["geometry"])

    print("Extracting buildings in city boundary...")
    buildings_in_city_boundaries = osm_buildings.get_buildings()

    print("Storing building data...")
    path_to_building_data = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}_buildings.pkl")

    with open(path_to_building_data, 'wb') as f:
        pickle.dump(buildings_in_city_boundaries, f)


def get_city_boundary(fp, city_name): #BUG extracts airport for Perth
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
    if type not in ["rgb", "irb", "buildings", "rgb_buildings", "r", "g", "b", "vnir", "stacked"]:
        print("Not a valid type.")
        return
    
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)

    path_to_image = os.path.join(path_to_city_data, f"{city_name}_{type}.png")
    image = mpimg.imread(path_to_image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    

def get_sentinel_image_data(temporal_extent, bands, city_name):
    print("Connecting to backend...")
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu/openeo/1.2").authenticate_oidc()

    global city_boundary
    min_longitude_point = city_boundary["geometry"].bounds[0]
    min_latitude_point = city_boundary["geometry"].bounds[1]
    max_longitude_point = city_boundary["geometry"].bounds[2]
    max_latitude_point = city_boundary["geometry"].bounds[3]

    """
    spatial_extent_test_img = {"west":13.294333, "south":52.454927, \
                    "east":13.500205, "north":52.574409} 
    """

    spatial_extent = {"west":min_longitude_point, "south":min_latitude_point, \
                      "east":max_longitude_point, "north":max_latitude_point} 

    datacube = connection.load_collection("SENTINEL2_L2A", \
            spatial_extent=spatial_extent, \
            temporal_extent=temporal_extent, \
            bands=bands
            )

    print("Downloading image data...")
    building_and_sentinel_data_dir_name = "building_and_sentinel_data"
    building_and_sentinel_city_data_dir_name = \
    f"{building_and_sentinel_data_dir_name}/{city_name}"
    path_to_image_data = os.path.join(building_and_sentinel_city_data_dir_name, f"{city_name}.tiff")

    datacube.download(path_to_image_data)

    src = rasterio.open(path_to_image_data)
    image_data = src.read()

    r_band = image_data[0]
    g_band = image_data[1]
    b_band = image_data[2]
    vnir_band = image_data[3]

    src.close()

    print("Processing image data...")
    r_band_clipped_norm = (np.clip(r_band, 0, 2500).astype(np.float64) / 2500) * 255
    g_band_clipped_norm = (np.clip(g_band, 0, 2500).astype(np.float64) / 2500) * 255
    b_band_clipped_norm = (np.clip(b_band, 0, 2500).astype(np.float64) / 2500) * 255
    vnir_band_clipped_norm = (np.clip(vnir_band, 0, 2500).astype(np.float64) / 2500) * 255

    rgb_im = np.stack([r_band_clipped_norm, g_band_clipped_norm, b_band_clipped_norm], axis=-1)
    irb_im = np.stack([vnir_band_clipped_norm, g_band_clipped_norm, b_band_clipped_norm], axis=-1)

    rgb_im = rgb_im.astype(np.uint8)
    irb_im = irb_im.astype(np.uint8)
    r_im = r_band_clipped_norm.astype(np.uint8)
    g_im = g_band_clipped_norm.astype(np.uint8)
    b_im = b_band_clipped_norm.astype(np.uint8)
    vnir_im = vnir_band_clipped_norm.astype(np.uint8)

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


def build_stacked_im(city_name, label_im, rgb_im):
    path_to_city_data = os.path.join("building_and_sentinel_data", city_name)
    mask = label_im == 1
    stacked_im = np.copy(rgb_im)
    stacked_im[mask] = [0, 0, 255] #set blue

    path_to_stacked_image = os.path.join(path_to_city_data, f"{city_name}_stacked.png")
    plt.imsave(path_to_stacked_image, stacked_im)

    buildings_im = np.zeros((rgb_im.shape[0], rgb_im.shape[1], 4), dtype=np.uint8)
    buildings_im[mask, 3] = 255
    buildings_im[mask, :3] = [0, 0, 255]

    path_to_buildings_image = os.path.join(path_to_city_data, f"{city_name}_buildings.png")
    plt.imsave(path_to_buildings_image, buildings_im)

def extract_coordinates(geometry):
    if geometry.geom_type == 'Polygon':
        exterior_coords = list(geometry.exterior.coords)
        return exterior_coords
    elif geometry.geom_type == 'MultiPolygon':
        all_coords = []
        for polygon in geometry.geoms:
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

    path_to_image_data = os.path.join(path_to_city_data, f"{city_name}.tiff")
    image_data = rasterio.open(path_to_image_data)

    path_to_building_data = os.path.join(path_to_city_data, f"{city_name}_buildings.pkl")
    buildings_data = None
    try:
        with open(path_to_building_data, 'rb') as f:
            buildings_data = pickle.load(f)
    except Exception as e:
        print(f"An error occurred: {e}")

    projected_building_data = buildings_data.to_crs(image_data.crs)
    transformer = rasterio.transform.AffineTransformer(image_data.transform)

    print("Generating labels...")
    projected_building_data['coords'] = projected_building_data['geometry'].apply(extract_coordinates)
    projected_building_data['row_col'] = projected_building_data['coords'].apply(lambda coords: coords_to_rowcol(coords, transformer))

    path_to_rgb_image = os.path.join(path_to_city_data, f"{city_name}_rgb.png")
    rgb_im = Image.open(path_to_rgb_image).convert('RGB')
    rgb_im = np.array(rgb_im)

    pixel_polygons = []
    for building in list(projected_building_data['row_col']):
        pixel_polygon = []
        for coord in building:
            x = coord[0]
            y = coord[1]
            pixel_coord = (y,x)
            pixel_polygon.append(pixel_coord)
        pixel_polygons.append(Polygon(pixel_polygon))

    label_im = rasterio.features.rasterize(pixel_polygons, out_shape=rgb_im.shape[:2], all_touched=True)

    print("Building stacked image...")
    build_stacked_im(city_name, label_im, rgb_im)


def a_1_pipeline(city_name):
    get_osm_building_data(city_name)

    temporal_extent=["2024-05-21", "2024-05-22"]
    bands=["B04", "B03", "B02", "B08"]
    get_sentinel_image_data(temporal_extent, bands, city_name)

    label_gen(city_name)

city_name = "Flensburg"
a_1_pipeline(city_name)
#plot_data(city_name, "buildings")