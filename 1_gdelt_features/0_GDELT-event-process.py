# Databricks notebook source
# MAGIC %md
# MAGIC This notebook cleans the Events dataset.  
# MAGIC 1. It harmonizes admin 1 names with those in ACLED.  
# MAGIC 2. It filters event types (cameo codes).  
# MAGIC 3. It stacks the Events dataset.  
# MAGIC 4. It places each row in the Events dataset in an admin 1 using lat / lon coordinates (+some cleaning).  

# COMMAND ----------

!pip install pysal
!pip install descartes

# COMMAND ----------

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime

from pyspark.sql.functions import coalesce, to_timestamp, to_date, lit
from pyspark.sql.types import FloatType

# COMMAND ----------

# country code
CO = 'UG'

# dates to filter
start_date = '2020-01-01'  # inclusive
end_date = '2023-05-01'  # exclusive: download does not include this day 

# cameo codes to filter in
# if no filter, set CAMEO_LST to None
CAMEO_LST = ['11','14','15','17','18','19','20']  

# COMMAND ----------

# database and table
DATABASE_NAME = 'news_media'
INPUT_TABLE_NAME = 'horn_africa_gdelt_events_brz'
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_events_cameo1_slv'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Harmonize admin 1 names

# COMMAND ----------

co_dict = {
    'DJ': {'CO_ACLED_NO': 97, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/djibouti_adm1/dji_admbnda_gadm_adm1_2022.shp'},  
    'ER': {'CO_ACLED_NO': 104, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/eritrea_adm1/eri_admbnda_adm1_gov_20200427.shp'},
    'ET': {'CO_ACLED_NO': 108, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/ethiopia_adm1/eth_admbnda_adm1_csa_bofedb_2021.shp'},
    'KE': {'CO_ACLED_NO': 175, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/kenya_adm1/ken_admbnda_adm1_iebc_20191031.shp'},
    'SO': {'CO_ACLED_NO': 224, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/somalia_adm1/som_admbnda_adm1_ocha_20230308.shp'},
    'OD': {'CO_ACLED_NO': 227, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/southsudan_adm1/ssd_admbnda_adm1_imwg_nbs_20221219.shp'},
    'SU': {'CO_ACLED_NO': 214, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/sudan_adm1/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp'},
    'UG': {'CO_ACLED_NO': 235, 'SHAPEFILE': '/dbfs/FileStore/geospatial/shapefiles/uganda_adm1/uga_admbnda_adm2_ubos_20200824.shp'} 
}


### Notes ###
# DJ: one admin 1 seems collapsed with capital
# UG: using admin level 2 shapefile to match ACLED (where admin 2 is under admin 1 col) 

if CO == 'UG':
    ADM = 'ADM2_EN'
else:
    ADM = 'ADM1_EN'

# COMMAND ----------

# the shapefile
gdf = gpd.read_file(co_dict[CO]['SHAPEFILE'])

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
d = d.filter((d['CountryFK'] == co_dict[CO]['CO_ACLED_NO']) & (d['TimeFK_Event_Date'] >= 20200101))
# standard admin values 
standard_admin = d.select('ACLED_Admin1').distinct().rdd.flatMap(list).collect()

# COMMAND ----------

######## CHECK HERE! ########
# check that admin names are in the standard name list 
new_admin = list(gdf[ADM])
not_in_acled = [x for x in new_admin if x not in standard_admin]
not_in_shp = [x for x in standard_admin if x not in new_admin]

# these need to be matched and fixed in the next cell
print(not_in_acled)
print(not_in_shp)

# COMMAND ----------

if (CO == 'UG') or (CO == 'OD') or (CO == 'SO') or (CO == 'ER'):
    pass
elif CO == 'DJ':
    gdf.loc[gdf['ADM1_EN']=='Djiboutii', 'ADM1_EN'] = 'Djibouti'
    gdf.loc[gdf['ADM1_EN']=='Tadjoura', 'ADM1_EN'] = 'Tadjourah'
elif CO == 'SU':
    # Note: 'Upper Nile', 'Bahr el Ghazal', 'Equatoria' are not in ACLED data after 2020
    gdf.loc[gdf['ADM1_EN']=='Abyei PCA', 'ADM1_EN'] = 'Abyei'
    gdf.loc[gdf['ADM1_EN']=='Aj Jazirah', 'ADM1_EN'] = 'Al Jazirah'
elif CO == 'KE':
    gdf.loc[gdf['ADM1_EN']=="Murang'a", 'ADM1_EN'] = 'Muranga'
    gdf.loc[gdf['ADM1_EN']=='Elgeyo-Marakwet', 'ADM1_EN'] = 'Elgeyo Marakwet'
elif CO == 'ET':
    gdf.loc[gdf['ADM1_EN']=='Benishangul Gumz', 'ADM1_EN'] = 'Benshangul/Gumuz'
    gdf.loc[gdf['ADM1_EN']=='South West Ethiopia', 'ADM1_EN'] = 'South West'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter events

# COMMAND ----------

# events data - filtered to date
events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")
events = events.withColumn('DATEADDED', to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
events = events.withColumn('DATEADDED', to_date('DATEADDED'))
events = events.filter((events['DATEADDED'] >= datetime.strptime(start_date, '%Y-%m-%d').date()) & (events['DATEADDED'] < datetime.strptime(end_date, '%Y-%m-%d').date()))

# filter by actors and location
events = events.filter((events.ActionGeo_CountryCode==CO) | (events.Actor1Geo_CountryCode==CO) | (events.Actor2Geo_CountryCode==CO))

print(events.count())

# COMMAND ----------

# filter to root events only 
events = events.filter(events.IsRootEvent=='1')

# filter by cameo codes
if CAMEO_LST is not None:
    events = events.filter(events.EventRootCode.isin(CAMEO_LST))

print(events.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stack events

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

# make data long in chunks (stack)
act1 = events.select('DATEADDED','SOURCEURL','Actor1Geo_FullName','Actor1Geo_ADM1Code','Actor1Geo_Lat','Actor1Geo_Long')
act2 = events.select('DATEADDED','SOURCEURL','Actor2Geo_FullName','Actor2Geo_ADM1Code','Actor2Geo_Lat','Actor2Geo_Long')
action = events.select('DATEADDED','SOURCEURL','ActionGeo_FullName','ActionGeo_ADM1Code','ActionGeo_Lat','ActionGeo_Long')
cols = ['DATEADDED','SOURCEURL','GEO_NAME','ADMIN1', 'LAT','LON']
act1 = act1.toDF(*cols)
act2 = act2.toDF(*cols)
action = action.toDF(*cols)
long_df = act1.union(act2).union(action)

#  keep only events in the country
long_df = long_df.fillna('', subset=['ADMIN1'])
long_df = long_df.filter(long_df['ADMIN1'].startswith(CO))
print(long_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Geo locate to admin 1 using lat / lon

# COMMAND ----------

# split into country-wide and admin-level df
co_df = long_df.filter(long_df['ADMIN1']==CO)
adm_df = long_df.filter(long_df['ADMIN1']!=CO)
print(co_df.count())
print(adm_df.count())

# some processing
co_df = co_df.drop(*['LAT','LON','GEO_NAME'])
co_df = co_df.withColumn('COUNTRY', lit(CO))
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
# this should only contain CO-wide code (like UG00)
# all else should be fixed in the next cell
not_placed = adm_gdf[pd.isna(adm_gdf[ADM])]['ADMIN1'].unique()
print(not_placed)  

for code in not_placed:
    print(code)
    print(adm_gdf[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']==code)]['GEO_NAME'].unique())  
    print()

# COMMAND ----------

# not place through coords AND admin1 --> fix
if CO == 'DJ':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='DJ07'), 'ADM1_EN'] = 'Djibouti'
    
elif CO == 'ER':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ER06'), 'ADM1_EN'] = 'Semienawi Keih Bahri'

elif CO == 'ET':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET53'), 'ADM1_EN'] = 'Tigray'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET49'), 'ADM1_EN'] = 'Gambela'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET47'), 'ADM1_EN'] = 'Benshangul/Gumuz'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET52'), 'ADM1_EN'] = 'Somali'

elif CO == 'KE':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='KE09'), 'ADM1_EN'] = 'Busia'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='KE07'), 'ADM1_EN'] = 'Migori'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='KE08'), 'ADM1_EN'] = 'Kajiado'

    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mtongwe, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mwache, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kilimani, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='English Point, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Vanga, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Diani Beach, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kombeni, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Shanzu, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Vuka, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kisite, Coast, Kenya'), 'ADM1_EN'] = 'Kwale' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Port Reitz, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Samaki, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Ungama Bay, Coast, Kenya'), 'ADM1_EN'] = 'Tana River'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kikambala, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Ishakani, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kiunga, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Shimanzi, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kinondo, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Tudor Creek, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Ras Kitau, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Englishpoint, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Portreitz, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Diani Beach Hotel, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kiangwe, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mkokoni, Coast, Kenya'), 'ADM1_EN'] = 'Lamu' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mandabay, Coast, Kenya'), 'ADM1_EN'] = 'Lamu' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Gazi Bay, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Shelly Beach, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Nyali Beach, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mvindeni, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Chale Island, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mtwapa Creek, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Jumba La Mtwana, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kizingoni, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Little Head, Coast, Kenya'), 'ADM1_EN'] = 'Lamu' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kilindini Harbour, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kilifi Creek, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Leisure Lodge, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mtongwe Ferry, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mkanda Channel, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Tenewi, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'

elif CO == 'SO':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO19'), 'ADM1_EN'] = 'Togdheer'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO03'), 'ADM1_EN'] = 'Bari'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO09'), 'ADM1_EN'] = 'Lower Juba'

elif CO == 'OD':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='OD07'), 'ADM1_EN'] = 'Upper Nile'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='OD09'), 'ADM1_EN'] = 'Western Bahr el Ghazal'

elif CO == 'SU':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SU36'), 'ADM1_EN'] = 'Red Sea'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SU49'), 'ADM1_EN'] = 'South Darfur'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SU47'), 'ADM1_EN'] = 'West Darfur'

elif CO == 'UG':
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UG34'), ADM] = 'Kabale'
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UGD6'), ADM] = 'Manafwa'
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UG43'), ADM] = 'Kisoro'
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UGC4'), ADM] = 'Bukwo'
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UG59'), ADM] = 'Ntungamo'
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UG28'), ADM] = 'Bundibugyo'
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UGC9'), ADM] = 'Isingiro'
    adm_gdf.loc[(pd.isna(adm_gdf[ADM])) & (adm_gdf['ADMIN1']=='UG61'), ADM] = 'Rakai'

# all other empty cells to be designated as the rest to entire country
adm_gdf[ADM].fillna(CO, inplace=True)

# COMMAND ----------

# processing
adm_gdf = adm_gdf[['DATEADDED','SOURCEURL', ADM]]
adm_gdf.columns = ['DATEADDED','SOURCEURL','ADMIN1']
adm_gdf = pd.concat([adm_gdf, pd.Series([CO for x in np.arange(adm_df.shape[0])], name='COUNTRY')], axis=1)  # add country column

# COMMAND ----------

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
