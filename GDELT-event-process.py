# Databricks notebook source
!pip install pysal
!pip install descartes

# COMMAND ----------

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np

from pyspark.sql.functions import coalesce, to_timestamp, to_date

# COMMAND ----------

######## CHANGE THIS! ########
# Process one country at a time
# CO = 'DJ' ### one admin 1 seems collapsed with capital
# CO_ACLED_NO = 97
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/djibouti_adm1/dji_admbnda_gadm_adm1_2022.shp'

# CO = 'ER'
# CO_ACLED_NO = 104
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/eritrea_adm1/eri_admbnda_adm1_gov_20200427.shp'

# CO = 'ET'
# CO_ACLED_NO = 108
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/ethiopia_adm1/eth_admbnda_adm1_csa_bofedb_2021.shp'

# CO = 'KE'
# CO_ACLED_NO = 175
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/kenya_adm1/ken_admbnda_adm1_iebc_20191031.shp'

# CO = 'SO'
# CO_ACLED_NO = 224
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/somalia_adm1/som_admbnda_adm1_ocha_20230308.shp'

# CO = 'OD'
# CO_ACLED_NO = 227
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/southsudan_adm1/ssd_admbnda_adm1_imwg_nbs_20221219.shp'

CO = 'SU'
CO_ACLED_NO = 214
SHAPEFILE = '/dbfs/FileStore/df/shapefiles/sudan_adm1/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp'

# CO = 'UG'
# CO_ACLED_NO = 235
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/uganda_adm1/uga_admbnda_adm1_ubos_20200824.shp'

# database and table
DATABASE_NAME = 'news_media'
INPUT_TABLE_NAME = 'horn_africa_gdelt_events_brz'
OUTPUT_TABLE_NAME = f'horn_africa_gdelt_events_{CO}_slv'

# COMMAND ----------

# events data
events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")

# COMMAND ----------

######## RATIONALE ########
# 1. each row in the events dataframe is an event linked to a source-url
# there can be multiple events in a source-url
# we filter to root events only to exclude peripheral events
# 2. each event has country locations for actor 1, actor 2, and action
# which can be null
# 3. we count an event as happening in-country 
# once each for actor 1, actor 2, and action
# this is to weight an event more if it is more relevant to the country
# e.g. if an event's actor 1 and action are sudan and actor 2 is ethiopia
# the event will count (be weighted) twice for sudan and once for ethiopia
# 4. note in the example that an event can be relevant to more than one country
# 5. in short, source-urls with more root events with more relevance to a country will be weighted more
###########################


# filter by actors and location
events = events.filter((events.ActionGeo_CountryCode==CO) | (events.Actor1Geo_CountryCode==CO) | (events.Actor2Geo_CountryCode==CO))
# filter to root events only 
events = events.filter(events.IsRootEvent=='1')

# make data long in chunks (stack)
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

######## METHOD ########
# 1. we locate each row in long_df to its admin 1 level through merges
# this is to avoid transforming pyspark to pandas and back again
# 2. we do this by creating a key-dataframe with
# admin 1 name, admin 1 code, and coordinates
########################

# make key of admin name and lat lon
key_df = long_df.dropDuplicates(['ADMIN1'])
key_df = key_df.toPandas()
key_df = key_df.loc[key_df['ADMIN1']!=CO, ['ADMIN1', 'LAT', 'LON']]

# COMMAND ----------

# shapefile
gdf = gpd.read_file(SHAPEFILE)

# COMMAND ----------

# merge the key-dataframe with shapefile to get admin 1 names
geometry = [Point(xy)  for xy in zip(key_df['LON'], key_df['LAT'])]
key_gdf = gpd.GeoDataFrame(key_df, crs=gdf.crs, geometry=geometry)
key_gdf = gpd.sjoin(gdf, key_gdf, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')
key_gdf = key_gdf.loc[:, ['ADM1_EN', 'ADMIN1']]

# COMMAND ----------

######## CHECK HERE! ########
# check that all admins are covered in the shapefile
not_in_shp = [k for k in list(key_df['ADMIN1']) if k not in list(key_gdf['ADMIN1'])]
assert len(not_in_shp) == 0, ('Some admins are not in the shapefile!')

# COMMAND ----------

not_in_shp

# COMMAND ----------

######## CHECK HERE! ########
# check that there are not duplicate admins
dup_adm = list(key_gdf[key_gdf.duplicated('ADM1_EN')]['ADM1_EN'])
assert len(dup_adm) == 0, ('Some admins are duplicated in the shapefile! check dup_adm!')

# COMMAND ----------

dup_adm

# COMMAND ----------

######## CHECK HERE! ########
# duplicated admins
key_gdf[key_gdf['ADM1_EN'].isin(dup_adm)]

# COMMAND ----------

######## CHANGE THIS! ########
problem_code = 'SU00'
# take out problem code from key-dataframe
key_gdf = key_gdf[key_gdf['ADMIN1'] != problem_code]

# COMMAND ----------

### problem codes are not classified correctly ###
### repeat the steps above to clean problem codes ###
stragglers = long_df.filter(long_df.ADMIN1 == problem_code)
stragglers = stragglers.toPandas()
stragglers = stragglers[['GEO_NAME', 'ADMIN1', 'LAT', 'LON']]
stragglers = stragglers.drop_duplicates()

# COMMAND ----------

# merge the straggler key-dataframe with shapefile
geometry = [Point(xy)  for xy in zip(stragglers['LON'], stragglers['LAT'])]
stragglers = gpd.GeoDataFrame(stragglers, crs=gdf.crs, geometry=geometry)
strg_key_gdf = gpd.sjoin(gdf, stragglers, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')
# more wrangling - we cannot have duplicate column names when merging in pyspark
strg_key_gdf = pd.merge(key_gdf, strg_key_gdf[['ADM1_EN','GEO_NAME']])
strg_key_gdf.drop('ADM1_EN', axis=1, inplace=True)
strg_key_gdf.columns = ['admin_1','geoname']
strg_key_gdf['adm1'] = problem_code

# COMMAND ----------

# convert to pyspark and merge into long_df
strg_key_gdf = sqlContext.createDataFrame(strg_key_gdf)
long_df = long_df.join(strg_key_gdf, (long_df.GEO_NAME==strg_key_gdf.geoname) & (long_df.ADMIN1==strg_key_gdf.adm1), how='left')

# COMMAND ----------

# processing / cleaning
long_df = long_df.withColumn('admin_1', coalesce(long_df.admin_1, long_df.ADMIN1)) 
long_df = long_df.drop(*['geoname', 'adm1', 'ADMIN1'])

# COMMAND ----------

# here we make sure that the admin 1 names match
# the outcome (ACLED) data

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

######## CHECK HERE! ########
# check that admin names are in the standard name list 
new_admin = list(key_gdf['ADM1_EN'])
wrong_name = [x for x in new_admin if x not in standard_admin]
assert len(wrong_name) == 0, 'Check admin names!'

# COMMAND ----------

######## CHECK HERE! ########
wrong_name

# COMMAND ----------

######## CHANGE THIS! ########
key_gdf.loc[key_gdf['ADM1_EN']=='Aj Jazirah', 'ADM1_EN'] = 'Al Jazirah'

# COMMAND ----------

# change column names and convert to pyspark to merge
key_gdf.columns = ['ADMIN1_NAME', 'adm1']
key_gdf = sqlContext.createDataFrame(key_gdf)

# COMMAND ----------

# merge for uniform admin names:)
# NOTE: admin name will be null for rows pertaining to the entire country
long_df = long_df.join(key_gdf, long_df['admin_1']==key_gdf['adm1'], how='left')

# processing / cleaning
long_df = long_df.drop(*['GEO_NAME', 'LAT', 'LON', 'adm1'])
long_df = long_df.withColumnRenamed('admin_1', 'ADMIN1_CODE')

# COMMAND ----------

display(long_df)

# COMMAND ----------

# save
long_df.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))
