# Databricks notebook source
!pip install rasterio
!pip install fiona
!pip install geopandas
!pip install rasterstats

# COMMAND ----------

import os
import rasterio
from rasterio.plot import show, show_hist
import matplotlib.pyplot as plt
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterstats import zonal_stats
import pandas as pd

# COMMAND ----------

DATABASE_NAME = 'news_media'
OUTPUT_TABLE_NAME = 'horn_africa_geo_popdens2020_static_slv'

# COMMAND ----------

target_country = ['djibouti', 'eritrea', 'ethiopia', 'kenya', 
                  'somalia', 'south_sudan', 'sudan', 'uganda']

# COMMAND ----------

# Get paths for the shape files and tiff files

def get_all_child_paths(parent_path, dir_path=False, file_path=False):
    """
    Fetch the child directory/file path from the given parent path

    Inputs:
        parent_path: string
        dir_path&file_path: bool indicating which type of path is to be fetched
    """
    child_paths = []

    for dirpath, dirnames, filenames in os.walk(parent_path):
        #For directory paths
        if dir_path:
            for dirname in dirnames:
                child_paths.append(os.path.join(dirpath, dirname))
        
        # For file paths
        if file_path:
            for filename in filenames:
                child_paths.append(os.path.join(dirpath, filename))

    return child_paths


shapefile_lst = get_all_child_paths('/dbfs/FileStore/df/shapefiles/', 
                                    dir_path=True)
tiff_lst = get_all_child_paths('/dbfs/FileStore/df/geotiff/', 
                               file_path=True)

# match the shape&tiff file combo with the corresponding country
shape_tiff = tuple(zip(shapefile_lst, tiff_lst))
country_file = dict(zip(target_country,shape_tiff))

# COMMAND ----------

country_file

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compatibility Check

# COMMAND ----------

def test_compatibility(shapefile_path, raster_path):
    """
    Function used to test compatibility of two files in terms of metadata

    Inputs:
        shapefile_path: the path of the shape file
        aster_path: the path of the raster file
    Outputs:
        Print statements specifying the result
    """
    
    print('***Results of Compatiblity Check:*** ')
    print('\n')
    
    with rasterio.open(raster_path) as raster:
        raster_crs = raster.crs
        raster_bounds = raster.bounds
        raster_res = raster.res
        raster_nodata = raster.nodata
        raster_dtype = raster.dtypes[0]
        
    with fiona.open(shapefile_path) as shapefile:
        shapefile_crs = shapefile.crs
        shapefile_schema = shapefile.schema
    
     # 1. Coordinate System / Projection
    if raster_crs == shapefile_crs:
        print("Coordinate systems are compatible!")
    else:
        print("Coordinate systems are NOT compatible!")
    print('\n')

    # 2. Spatial Extent
    gdf = gpd.read_file(shapefile_path)
    overlaps = True

    if gdf.total_bounds[0] < raster_bounds.left:
        print("The left boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if gdf.total_bounds[1] < raster_bounds.bottom:
        print("The bottom boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if gdf.total_bounds[2] > raster_bounds.right:
        print("The right boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if gdf.total_bounds[3] > raster_bounds.top:
        print("The top boundary of the shapefile is outside the raster's extent.")
        overlaps = False

    if overlaps:
        print("Spatial extents are compatible!")
    else:
        print("Spatial extents do not perfectly overlap!")
    print('\n')
        
    # 3. Resolution (for raster)
    print(f"Raster resolution: {raster_res}")
    print('\n')
    
    # 4. NoData Value (for raster)
    print(f"NoData value for the raster: {raster_nodata}")
    print('\n')
    
    # 5. Attribute Data (for vector)
    print("Shapefile attributes:")
    print(shapefile_schema)
    print('\n')
    
    # 6. Data Types
    print(f"Raster data type: {raster_dtype}")  
    print(f"Vector data types: {shapefile_schema['properties']}")
    print('\n')
    

def geo_plot(shapefile_path, raster_path, hist=False):
    """
    Give an initial plot of the combination of the shape file and the ratser file
    
    Inputs:
        shapefile_path: the path of the shape file
        aster_path: the path of the raster file
        hist(bool): check if a histogram is needed
    Outputs:
        A plot
    """
    
    shape_gdf = gpd.read_file(shapefile_path)
    raster_file = rasterio.open(raster_path, mode='r')
    
    if hist:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
        show(raster_file, ax=ax1, title="Population Density 2020")
        shape_gdf.plot(ax=ax1, facecolor="None", edgecolor="yellow")
        show_hist(raster_file, ax=ax2, title="Histogram of Population Density")
    else:
        fig, ax1 = plt.subplots(1,1)
        show(raster_file, ax=ax1, title="Population Density 2020")
        shape_gdf.plot(ax=ax1, facecolor="None", edgecolor="yellow")
    
    plt.show()

# COMMAND ----------

shapefile_path, raster_path = country_file['uganda']
test_compatibility(shapefile_path, raster_path)

# COMMAND ----------

geo_plot(shapefile_path, raster_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Merge

# COMMAND ----------

def merge_geo(shapefile_path, raster_path):
    """
    Merge two datasets by calculating the zonal statistics
    
    Inputs:
        shapefile_path: the path of the shape file
        aster_path: the path of the raster file
    Outputs:
        stats: a dict of lists containing relevant stats
    """

    # Calculate the zonal stats
    stats = zonal_stats(shapefile_path, raster_path)
    
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(shapefile_path)
    
    # Convert the list of dictionaries to a DataFrame
    stats_df = pd.DataFrame(stats)

    # Join the zonal statistics DataFrame with the original GeoDataFrame
    gdf_stats = gdf.join(stats_df)
    
    return gdf_stats

# COMMAND ----------

df = pd.DataFrame()

co_dict = {
    'djibouti':'DJ', 
    'eritrea':'ER', 
    'ethiopia':'ET', 
    'kenya':'KE', 
    'somalia':'SO',
    'south_sudan':'OD', 
    'sudan':'SU', 
    'uganda':'UG'
}

for country, files in country_file.items():
    shapefile_path, raster_path = files
    gdf = merge_geo(shapefile_path, raster_path)
    if 'ADM2_EN' not in gdf.columns:
        gdf.loc[:, 'ADM2_EN'] = None
    gdf_stats = gdf.loc[:,['ADM1_EN', 'ADM2_EN', 'mean']]
    gdf_stats.loc[:, 'COUNTRY'] = co_dict[country]
    df = pd.concat([df, gdf_stats], axis=0)

df['ADMIN1'] =  df.apply(lambda row: row['ADM1_EN'] if row['ADM2_EN']==None else row['ADM2_EN'], axis=1)
df.rename(columns={'mean': 'mean_pop_dense_2020'}, inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(['ADM1_EN','ADM2_EN'], axis=1, inplace=True)

# COMMAND ----------

df

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df = spark.createDataFrame(df)

# COMMAND ----------

# Save the data 
df.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))

# COMMAND ----------


