# Databricks notebook source
!pip install pysal
!pip install descartes

# COMMAND ----------

from shapely.geometry import Point
import geopandas as gpd
import pandas as pd
import numpy as np
# pd.set_option('display.max_rows', None)

# COMMAND ----------

DATABASE_NAME = 'news_media'
INPUT_TABLE_NAME = 'horn_africa_gdelt_events_brz'
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_brz'

CO = 'SU'
CO_ACLED_NO = 214

# COMMAND ----------

events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")

# COMMAND ----------

# Process one country at a time
# Note: an article can be used for more than one country


# filter by actors and location
events = events.filter((events.ActionGeo_CountryCode==CO) | (events.Actor1Geo_CountryCode==CO) | (events.Actor2Geo_CountryCode==CO))
# filter to root events only (ignore peripheral events)
events = events.filter(events.IsRootEvent=='1')

# filter date for now
# import datetime as dt
# import pyspark.sql.functions as F
# events = events.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
# events = events.withColumn('DATEADDED', F.to_date('DATEADDED'))
# events = events.filter(events.DATEADDED < dt.date(2020,4,1))

# make data long 
# an event can be counted up to 3 times (once each for actor 1, actor 2 and action)
# rationale - 
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

# from pyspark.sql.functions import udf
# from pyspark.sql.types import StringType

# def fix_adm(adm1, geo, problem_code=problem_code, ky_df=strg_key_gdf):
#     if adm1 == problem_code:
#         new_adm1 = ky_df.loc[ky_df['GEO_NAME']==geo, 'ADMIN1'].iloc[0]
#         return new_adm1
#     else:
#         return adm1
    
# fix_adm_udf = udf(fix_adm, StringType())
# long_df = long_df.withColumn('ADMIN1', fix_adm_udf(long_df['ADMIN1'], long_df['GEO_NAME']))

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

display(ldf)

# COMMAND ----------

# merge names back in
long_df = pd.merge(long_df, key_df, on='ADMIN1', how='left')

# COMMAND ----------

long_df['ADM1_EN'] = long_df['ADM1_EN'].fillna(CO)

# COMMAND ----------

embed = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{OUTPUT_TABLE_NAME}")

# COMMAND ----------

# merge with embeddings
all_df = pd.merge(long_df, embed, left_on='SOURCEURL', right_on='url', how='left')

# COMMAND ----------

embed_multiples = all_df.loc[:, ['SQLDATE', 'ADM1_EN']+list(np.arange(512).astype(str))]

# COMMAND ----------

import datetime as dt
embed_multiples['SQLDATE'] = embed_multiples['SQLDATE'].apply(lambda x: dt.datetime.strptime(str(x),'%Y%m%d'))

# COMMAND ----------

embed_multiples = embed_multiples[embed_multiples['SQLDATE'] >= dt.datetime(2020,1,1)]

# COMMAND ----------

freq = '3D'
co_frac = embed_multiples.groupby([pd.Grouper(key='SQLDATE', freq=freq), 'ADM1_EN']).size() / embed_multiples.groupby([pd.Grouper(key='SQLDATE', freq=freq)]).size()
idx = pd.IndexSlice
co_fracs = pd.DataFrame(co_frac).loc[idx[:, CO], :]
co_fracs

# COMMAND ----------

# cast as category to have all admin 1s even if nan
embed_multiples['ADM1_EN'] = embed_multiples['ADM1_EN'].astype('category')

# COMMAND ----------

# average by date and admin 1
avg = embed_multiples.groupby([pd.Grouper(key='SQLDATE', freq=freq), 'ADM1_EN']).mean()

# COMMAND ----------

# unique dates and admin 1s
dates = avg.index.get_level_values(0).unique()
admins = avg.index.get_level_values(1).unique()
admins = [a for a in admins if a != CO]

# COMMAND ----------

# cycle through and replace with weighted averages
for d in dates:
    for a in admins:
        co_per = co_fracs.loc[idx[d,CO], 0]
        adm_per = 1 - co_per
        if avg.loc[idx[(d, a)], :].isnull().values.any():
            # if NaN, repace with country's news
            avg.loc[idx[d, a], :] = avg.loc[idx[d, CO], :]
        else:
            # if not, take a weighted average
            avg.loc[idx[d, a], :] = np.average(avg.loc[idx[(d, [a, CO])], :], weights=[adm_per, co_per], axis=0)


# COMMAND ----------

# drop CO rows
avg = avg.loc[idx[:, admins], :]
avg 

# COMMAND ----------


