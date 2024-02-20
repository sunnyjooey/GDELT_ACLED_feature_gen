# Databricks notebook source
!pip install rasterio
!pip install fiona
!pip install geopandas
!pip install rasterstats

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')
from db_table import DATABASE_NAME, GEO_POP_DENSE_TABLE

sys.path.append('./geo_util')
from g_util import test_compatibility, geo_plot

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

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compatibility Check

# COMMAND ----------

# check uganda
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

# convert
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df = spark.createDataFrame(df)

# COMMAND ----------

# save in delta lake
# this will write if the table does not exist, but throw an error if it does exist
df.write.mode('errorifexists').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, GEO_POP_DENSE_TABLE))

# COMMAND ----------


