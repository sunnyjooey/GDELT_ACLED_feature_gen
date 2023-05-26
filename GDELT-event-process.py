# Databricks notebook source
!pip install pysal
!pip install descartes

# COMMAND ----------

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np

from pyspark.sql.functions import coalesce, to_timestamp, to_date
from pyspark.sql.types import FloatType

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

CO = 'SO'
CO_ACLED_NO = 224
SHAPEFILE = '/dbfs/FileStore/df/shapefiles/somalia_adm1/som_admbnda_adm1_ocha_20230308.shp'

# CO = 'OD'
# CO_ACLED_NO = 227
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/southsudan_adm1/ssd_admbnda_adm1_imwg_nbs_20221219.shp'

# CO = 'SU'
# CO_ACLED_NO = 214
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/sudan_adm1/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp'

# CO = 'UG'
# CO_ACLED_NO = 235
# SHAPEFILE = '/dbfs/FileStore/df/shapefiles/uganda_adm1/uga_admbnda_adm1_ubos_20200824.shp'

# database and table
DATABASE_NAME = 'news_media'
INPUT_TABLE_NAME = 'horn_africa_gdelt_events_brz'
OUTPUT_TABLE_NAME = f'horn_africa_gdelt_events_{CO}_slv'

# COMMAND ----------

### sort out admin 1 names first ###
# the shapefile
gdf = gpd.read_file(SHAPEFILE)

# COMMAND ----------

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
new_admin = list(gdf['ADM1_EN'])
not_in_acled = [x for x in new_admin if x not in standard_admin]
not_in_shp = [x for x in standard_admin if x not in new_admin]
# assert len(not_in_acled) == 0, 'Check admin names!'

# COMMAND ----------

print(not_in_acled)
print(not_in_shp)

# COMMAND ----------

######## CHANGE THIS! ########
# gdf.loc[gdf['ADM1_EN']=="Murang'a", 'ADM1_EN'] = 'Muranga'
# gdf.loc[gdf['ADM1_EN']=='Elgeyo-Marakwet', 'ADM1_EN'] = 'Elgeyo Marakwet'

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
act1 = events.select('DATEADDED','SOURCEURL','Actor1Geo_FullName','Actor1Geo_ADM1Code','Actor1Geo_Lat','Actor1Geo_Long')
act2 = events.select('DATEADDED','SOURCEURL','Actor2Geo_FullName','Actor2Geo_ADM1Code','Actor2Geo_Lat','Actor2Geo_Long')
action = events.select('DATEADDED','SOURCEURL','ActionGeo_FullName','ActionGeo_ADM1Code','ActionGeo_Lat','ActionGeo_Long')
cols = ['DATEADDED','SOURCEURL','GEO_NAME','ADMIN1', 'LAT','LON']
act1 = act1.toDF(*cols)
act2 = act2.toDF(*cols)
action = action.toDF(*cols)
long_df = act1.union(act2).union(action)
long_df = long_df.withColumn('DATEADDED', to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
long_df = long_df.withColumn('DATEADDED', to_date('DATEADDED'))

#  keep only events in the country
long_df = long_df.fillna('', subset=['ADMIN1'])
long_df = long_df.filter(long_df['ADMIN1'].startswith(CO))
print(long_df.count())

# COMMAND ----------

# split into country-wide and admin-level df
co_df = long_df.filter(long_df['ADMIN1']==CO)
adm_df = long_df.filter(long_df['ADMIN1']!=CO)
print(co_df.count())
print(adm_df.count())

# some processing
co_df = co_df.drop(*['LAT','LON','GEO_NAME'])
adm_df = adm_df.withColumn('LON', adm_df['LON'].cast(FloatType()))
adm_df = adm_df.withColumn('LAT', adm_df['LAT'].cast(FloatType()))

# convert to pandas for merge
adm_df = adm_df.toPandas()

# COMMAND ----------

# merge the data with shapefile to get admin 1 names
geometry = [Point(xy)  for xy in zip(adm_df['LON'], adm_df['LAT'])]
adm_gdf = gpd.GeoDataFrame(adm_df, crs=gdf.crs, geometry=geometry)
adm_gdf = gpd.sjoin(adm_gdf, gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
print(adm_gdf.shape)

# COMMAND ----------

#### CHECK HERE ####
adm_gdf[pd.isna(adm_gdf['ADM1_EN'])]['ADMIN1'].unique()

# COMMAND ----------

adm_gdf[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO09')]['GEO_NAME'].unique()

# COMMAND ----------

#### RUN IF NEEDED ####
# not place through coords AND admin1 --> fix
adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO19'), 'ADM1_EN'] = 'Togdheer'
adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO03'), 'ADM1_EN'] = 'Bari'
adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO09'), 'ADM1_EN'] = 'Lower Juba'

# designate the rest to entire country
adm_gdf['ADM1_EN'].fillna(CO, inplace=True)

# COMMAND ----------

# processing
adm_gdf = adm_gdf[['DATEADDED','SOURCEURL','ADM1_EN']]
adm_gdf.columns = ['DATEADDED','SOURCEURL','ADMIN1']

# convert back to spark
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
adm_gdf = spark.createDataFrame(adm_gdf)

# COMMAND ----------

# combine country-wide and admin-level df back together
long_df = co_df.union(adm_gdf)
long_df.count()

# COMMAND ----------

# save
long_df.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))

# COMMAND ----------


