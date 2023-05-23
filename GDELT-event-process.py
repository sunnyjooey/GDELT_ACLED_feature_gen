# Databricks notebook source
!pip install pysal
!pip install descartes

# COMMAND ----------

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np

# COMMAND ----------

# database and table
DATABASE_NAME = 'news_media'
INPUT_TABLE_NAME = 'horn_africa_gdelt_events_brz'
OUTPUT_TABLE_NAME = ''

# Process one country at a time
CO = 'SU'
CO_ACLED_NO = 214

# COMMAND ----------

events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")

# COMMAND ----------

######## NEED NOTES HERE ON RATIONALE ############

# filter by actors and location
events = events.filter((events.ActionGeo_CountryCode==CO) | (events.Actor1Geo_CountryCode==CO) | (events.Actor2Geo_CountryCode==CO))
# filter to root events only (ignore peripheral events)
events = events.filter(events.IsRootEvent=='1')

# make data long 
act1 = events.select('SQLDATE','SOURCEURL','Actor1Geo_FullName','Actor1Geo_ADM1Code','Actor1Geo_Lat','Actor1Geo_Long')
act2 = events.select('SQLDATE','SOURCEURL','Actor2Geo_FullName','Actor2Geo_ADM1Code','Actor2Geo_Lat','Actor2Geo_Long')
action = events.select('SQLDATE','SOURCEURL','ActionGeo_FullName','ActionGeo_ADM1Code','ActionGeo_Lat','ActionGeo_Long')
cols = ['SQLDATE','SOURCEURL','GEO_NAME','ADMIN1', 'LAT','LON']
act1 = act1.toDF(*cols)
act2 = act2.toDF(*cols)
action = action.toDF(*cols)
long_df = act1.union(act2).union(action)

#  keep only events in the country
long_df = long_df.fillna('', subset=['ADMIN1'])
long_df = long_df.filter(long_df.ADMIN1.startswith(CO))
print(long_df.count())

# COMMAND ----------

# make key of admin name and lat lon
key_df = long_df.dropDuplicates(['ADMIN1'])
key_df = key_df.toPandas()
key_df = key_df.loc[key_df['ADMIN1']!=CO, ['ADMIN1', 'LAT', 'LON']]

# COMMAND ----------

# shapefile
gdf = gpd.read_file('/dbfs/FileStore/df/shapefiles/sudan_adm1/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp')

# COMMAND ----------

# Define geometry of events data
geometry = [Point(xy)  for xy in zip(key_df['LON'], key_df['LAT'])]
# Build spatial data frame
key_gdf = gpd.GeoDataFrame(key_df, crs=gdf.crs, geometry=geometry)
# Create merged spatial data frame to confirm matching dimensions
key_gdf = gpd.sjoin(gdf, key_gdf, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

# COMMAND ----------

# only columns needed
key_gdf = key_gdf.loc[:, ['ADM1_EN', 'ADMIN1']]

# COMMAND ----------

# check that all admins are covered in the shapefile
assert len([k for k in list(key_df['ADMIN1']) if k not in list(key_gdf['ADMIN1'])]) == 0, ('Some admins are not in the shapefile!')

# COMMAND ----------

# check that there are not duplicate admins
dup_adm = list(key_gdf[key_gdf.duplicated('ADM1_EN')]['ADM1_EN'])
assert len(dup_adm) == 0, ('Some admins are duplicated in the shapefile! check dup_adm')

# COMMAND ----------

# duplicated admins
key_gdf[key_gdf['ADM1_EN'].isin(dup_adm)]

# COMMAND ----------

# change this!
problem_code = 'SU00'
# take out problem code
key_gdf = key_gdf[key_gdf['ADMIN1'] != problem_code]

# COMMAND ----------

# problem_code are not classified correctly - fix here
stragglers = long_df.filter(long_df.ADMIN1 == problem_code)
stragglers = stragglers.toPandas()
stragglers = stragglers[['GEO_NAME', 'ADMIN1', 'LAT', 'LON']]
stragglers = stragglers.drop_duplicates()

# COMMAND ----------

# Define geometry of events data
geometry = [Point(xy)  for xy in zip(stragglers['LON'], stragglers['LAT'])]
# Build spatial data frame
stragglers = gpd.GeoDataFrame(stragglers, crs=gdf.crs, geometry=geometry)
# Create merged spatial data frame to confirm matching dimensions
strg_key_gdf = gpd.sjoin(gdf, stragglers, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

# COMMAND ----------

strg_key_gdf = pd.merge(key_gdf, strg_key_gdf[['ADM1_EN','GEO_NAME']])
strg_key_gdf.drop('ADM1_EN', axis=1, inplace=True)
strg_key_gdf.columns = ['admin_1','geoname']
strg_key_gdf['adm1'] = problem_code

# COMMAND ----------

strg_key_gdf

# COMMAND ----------

strg_key_gdf = sqlContext.createDataFrame(strg_key_gdf)

# COMMAND ----------

long_df = long_df.join(strg_key_gdf, (long_df.GEO_NAME==strg_key_gdf.geoname) & (long_df.ADMIN1==strg_key_gdf.adm1), how='left')

# COMMAND ----------

from pyspark.sql.functions import coalesce

long_df = long_df.withColumn('admin_1', coalesce(long_df.admin_1, long_df.ADMIN1)) 

# COMMAND ----------

long_df = long_df.drop(*['geoname', 'adm1', 'ADMIN1'])

# COMMAND ----------

display(long_df)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

database_host = dbutils.secrets.get(scope='warehouse_scope', key='database_host')
database_port = dbutils.secrets.get(scope='warehouse_scope', key='database_port')
user = dbutils.secrets.get(scope='warehouse_scope', key='user')
password = dbutils.secrets.get(scope='warehouse_scope', key='password')

database_name = "UNDP_DW_CRD"
table = "dbo.CRD_ACLED"
url = f"jdbc:sqlserver://{database_host}:{database_port};databaseName={database_name};"

d = (spark.read
      .format("com.microsoft.sqlserver.jdbc.spark")
      .option("url", url)
      .option("dbtable", table)
      .option("user", user)
      .option("password", password)
      .load()
    )

# filter to sudan    
d = d.filter(d['CountryFK']==CO_ACLED_NO)
# standard admin values 
standard_admin = d.select('ACLED_Admin1').distinct().rdd.flatMap(list).collect()

# COMMAND ----------

# check that admin names are in standard name list or change dictionary
new_admin = list(key_gdf['ADM1_EN'])
wrong_name = [x for x in new_admin if x not in standard_admin]
assert len(wrong_name) == 0, 'Check admin names!'

# COMMAND ----------

# check
wrong_name

# COMMAND ----------

# change manually -- change here!
key_gdf.loc[key_gdf['ADM1_EN']=='Aj Jazirah', 'ADM1_EN'] = 'Al Jazirah'

# COMMAND ----------

from pyspark.sql.types import StringType, StructField, StructType

# change column names before merging
key_gdf.columns = ['ADMIN1_NAME', 'adm1']
key_gdf = sqlContext.createDataFrame(key_gdf)

# COMMAND ----------

display(key_gdf)

# COMMAND ----------

long_df = long_df.join(key_gdf, long_df['admin_1']==key_gdf['adm1'], how='left')

# COMMAND ----------

long_df = long_df.drop(*['GEO_NAME', 'LAT', 'LON', 'adm1'])

# COMMAND ----------

long_df = long_df.withColumnRenamed('admin_1', 'ADMIN1_CODE')

# COMMAND ----------

display(long_df)

# COMMAND ----------


