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
INPUT_TABLE_NAME = 'sudan_gdelt_events_brz'
OUTPUT_TABLE_NAME = 'sudan_gdelt_gsgembed_brz'

CO = 'SU'

# COMMAND ----------

events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")
embed = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{OUTPUT_TABLE_NAME}")

# COMMAND ----------

events = events.toPandas()
embed = embed.toPandas()

# COMMAND ----------

# filter to root events only
events = events[events['IsRootEvent']==1]

# COMMAND ----------

# make data long
act1 = events.loc[:,['SQLDATE','SOURCEURL','Actor1Geo_FullName','Actor1Geo_ADM1Code','Actor1Geo_Lat','Actor1Geo_Long']]
act2 = events.loc[:,['SQLDATE','SOURCEURL','Actor2Geo_FullName','Actor2Geo_ADM1Code','Actor2Geo_Lat','Actor2Geo_Long']]
action = events.loc[:,['SQLDATE','SOURCEURL','ActionGeo_FullName','ActionGeo_ADM1Code','ActionGeo_Lat','ActionGeo_Long']]
cols = ['SQLDATE','SOURCEURL','GEO_NAME','ADMIN1', 'LAT','LON']
act1.columns = cols
act2.columns = cols
action.columns = cols
long_df = pd.concat([act1, act2, action])

# keep only events in sudan
long_df['ADMIN1'].fillna('', inplace=True)
long_df['keep'] = False
long_df.loc[long_df['ADMIN1'].str.startswith(CO), 'keep'] = True
long_df = long_df.loc[long_df['keep'], :]
long_df.drop('keep', axis=1, inplace=True)

# COMMAND ----------

# make key of admin name and lat lon
key_df = long_df.drop_duplicates('ADMIN1')
key_df = key_df.loc[key_df['ADMIN1']!='SU',['ADMIN1', 'LAT', 'LON']]

# COMMAND ----------

# shapefile
gdf = gpd.read_file('/dbfs/FileStore/df/shapefiles/sudan_adm1/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp')

# COMMAND ----------

# Define geometry of events data
geometry = [Point(xy)  for xy in zip(key_df['LON'], key_df['LAT'])]
# Build spatial data frame
key_df = gpd.GeoDataFrame(key_df, crs=gdf.crs, geometry=geometry)
# Create merged spatial data frame to confirm matching dimensions
key_df = gpd.sjoin(gdf, key_df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

# COMMAND ----------

# only columns needed
key_df = key_df.loc[:, ['ADM1_EN', 'ADMIN1']]
key_df

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
d = d.filter(d['CountryFK']==214)
# standard admin values 
standard_admin = d.select('ACLED_Admin1').distinct().rdd.flatMap(list).collect()

# COMMAND ----------

# check that admin names are in standard name list or change dictionary
change_dict = {'Aj Jazirah':'Al Jazirah', 'Abyei PCA': 'Abyei'}
new_admin = list(key_df['ADM1_EN'])
wrong_name = [x for x in new_admin if x not in standard_admin + list(change_dict.keys())]
assert len(wrong_name) == 0, 'Check admin names!'

# COMMAND ----------

# make admin names uniform (to the ones in ACLED)
key_df['ADM1_EN'] = key_df['ADM1_EN'].apply(lambda x: change_dict[x] if x in change_dict.keys() else x)

# COMMAND ----------

# merge names back in
long_df = pd.merge(long_df, key_df, on='ADMIN1', how='left')

# COMMAND ----------

long_df['ADM1_EN'] = long_df['ADM1_EN'].fillna(CO)

# COMMAND ----------

# merge with embeddings
all_df = pd.merge(long_df, embed, left_on='SOURCEURL', right_on='url', how='left')

# COMMAND ----------

embed_multiples = all_df.loc[:, ['SQLDATE', 'ADM1_EN']+list(np.arange(512).astype(str))]

# COMMAND ----------

import datetime as dt
embed_multiples['SQLDATE'] = embed_multiples['SQLDATE'].apply(lambda x: dt.datetime.strptime(str(x),'%Y%m%d'))

# COMMAND ----------

embed_multiples = embed_multiples[embed_multiples['SQLDATE'] >= dt.datetime(2023,1,1)]

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


