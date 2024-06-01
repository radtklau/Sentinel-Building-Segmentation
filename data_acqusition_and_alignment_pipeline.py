from pyrosm.data import sources
from pyrosm import get_data
from tqdm import tqdm
from pyrosm import OSM
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import os

def get_osm_city_data(city_name):
    osm_data_dir_name = "city_osm_data"
    if not os.path.exists(osm_data_dir_name):
        os.makedirs(osm_data_dir_name)
    
    try:
        fp = get_data(city_name, directory=osm_data_dir_name)
    except Exception as e:
        print(f"An error occurred: {e}")

    osm = OSM(fp)
    boundaries = osm.get_boundaries()
    boundaries_city = boundaries[boundaries['name']==city_name]


