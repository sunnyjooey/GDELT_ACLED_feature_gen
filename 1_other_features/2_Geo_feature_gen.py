# Databricks notebook source
!pip install rasterio
!pip install fiona
!pip install geopandas
!pip install rasterstats

# COMMAND ----------

dbutils.library.restartPython()

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

# import variables
import sys
sys.path.append('../util')

from db_table import DATABASE_NAME, GEO_POP_DENSE_TABLE

# COMMAND ----------

# country mapping
target_country = {'djibouti':('dji', 'DJ'), 
                  'eritrea':('eri', 'ER'), 
                  'ethiopia':('eth', 'ET'), 
                  'kenya':('ken', 'KE'), 
                  'somalia':('som', 'SO'), 
                  'southsudan':('ssd', 'OD'), 
                  'sudan':('sdn', 'SU'), 
                  'uganda':('uga', 'UG')}

# file paths
shapefile_path = '/dbfs/FileStore/geospatial/shapefiles/'
tiff_path = '/dbfs/FileStore/geospatial/geotiff/hoa_pop_dense/'

# COMMAND ----------

# match shape and tiff files
shape_dirs = os.listdir(shapefile_path)
tiff_files = next(os.walk(tiff_path), (None, None, []))[2]

country_file = {}
for country, abv in target_country.items():
    if country == 'sudan':
        shp = [dr for dr in shape_dirs if (country in dr) and ('south' not in dr)]
    else:
        shp = [dr for dr in shape_dirs if country in dr]
    tiff = [fl for fl in tiff_files if abv[0] in fl]
    country_file[country] = (os.path.join(shapefile_path, shp[0]), os.path.join(tiff_path, tiff[0]))

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

# for uniform admin names
adm_dict = {
    'Aj Jazirah':'Al Jazirah',
    'Abyei PCA':'Abyei',
    'Djiboutii':'Djibouti',
    'Benishangul Gumz':'Benshangul/Gumuz',
    "Murang'a":'Muranga',
    'Tadjoura':'Tadjourah',
    'South West Ethiopia':'South West',
    'Elgeyo-Marakwet':'Elgeyo Marakwet'
}

# Get stats by admin - this is where it happens
df = pd.DataFrame()
for country, files in country_file.items():
    shapefile_path, raster_path = files
    gdf = merge_geo(shapefile_path, raster_path)
    if 'ADM2_EN' not in gdf.columns:
        gdf.loc[:, 'ADM2_EN'] = None
    gdf_stats = gdf.loc[:,['ADM1_EN', 'ADM2_EN', 'mean']]
    gdf_stats.loc[:, 'COUNTRY'] = target_country[country][1]
    df = pd.concat([df, gdf_stats], axis=0)

# Some data cleaning 
df['ADMIN1'] =  df.apply(lambda row: row['ADM1_EN'] if row['ADM2_EN']==None else row['ADM2_EN'], axis=1)
df['ADMIN1'] = df.apply(lambda row: adm_dict[row['ADMIN1']] if row['ADMIN1'] in adm_dict else row['ADMIN1'], axis=1)
df.rename(columns={'mean': 'mean_pop_dense_2020'}, inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(['ADM1_EN','ADM2_EN'], axis=1, inplace=True)

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df = spark.createDataFrame(df)

# COMMAND ----------

# save in delta lake
# this will write if the table does not exist, but throw an error if it does exist
df.write.mode('errorifexists').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, GEO_POP_DENSE_TABLE))

# COMMAND ----------


