# Databricks notebook source
import numpy as np
import pandas as pd
import datetime as dt
import functools
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
CO = dbutils.widgets.get("CO")
print(CO)

# COMMAND ----------

co = CO.lower()
DATABASE_NAME = 'news_media'
EVTSLV_TABLE_NAME = f'horn_africa_gdelt_events_a1_slv'
EMB_TABLE_NAME = 'horn_africa_gdelt_gsgembed_brz'
OUTPUT_TABLE_NAME = ''

# COMMAND ----------

# readin data
evtslv = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVTSLV_TABLE_NAME}")
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EMB_TABLE_NAME}")

# some cleaning
evtslv = evtslv.filter(evtslv['DATEADDED'] >= dt.date(2020, 1, 1))
emb = emb.drop('DATEADDED')

# COMMAND ----------

# merge events and embeddings
all_df = evtslv.join(emb, evtslv.SOURCEURL==emb.url, how='left')
cols = ['DATEADDED', 'ADMIN1'] + list(np.arange(512).astype(str))
all_df = all_df.select(*cols)

# COMMAND ----------

# n = all_df.groupBy(F.window(F.col("DATEADDED"), "1 week"), 'ADMIN1').agg(F.count(F.lit(1)).alias("nom"))
# d = all_df.groupBy(F.window(F.col("DATEADDED"), "1 week")).agg(F.count(F.lit(1)).alias("denom"))
# d = n.join(d, 'window').withColumn('admin_frac', F.col('nom')/F.col('denom'))
# display(d)

# COMMAND ----------

# get average embeddings by time intervals (2-week) and admin 1
m = all_df.groupBy(F.window(F.col("DATEADDED"), "2 week", "1 week", "-3 day"), 'ADMIN1').mean()
cols = ['DATE', 'ADMIN1'] + list(np.arange(512).astype(str))
m = m.toDF(*cols)

# cycle through admin 1s
admins = m.select('ADMIN1').distinct().rdd.flatMap(list).collect()
admins = [a for a in admins if a != CO]

# average admin 1 and whole country embedding
# this takes whole country embedding if admin 1 is missing
collect_dfs = []
for admin in admins:
    adm = m.filter((m.ADMIN1==admin) | (m.ADMIN1==CO))
    adm = adm.groupby(F.col('DATE')).mean()
    cols = ['DATE'] + list(np.arange(512).astype(str))
    adm = adm.toDF(*cols)
    adm = adm.withColumn('ADMIN1', F.lit(admin))
    collect_dfs.append(adm) 

# collapse into one df and clean
df = functools.reduce(DataFrame.union, collect_dfs)
df = df.withColumn('STARTDATE', F.to_date(df['DATE']['start']))
df = df.withColumn('ENDDATE', F.to_date(df['DATE']['end']))
cols = ['STARTDATE', 'ENDDATE', 'ADMIN1'] + list(np.arange(512).astype(str))
df = df.select(*cols)
print(df.count())

# COMMAND ----------

# save
df.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))

# COMMAND ----------

# ### go back and save in one dataframe to decrease the number of tables
# from functools import reduce
# from pyspark.sql import DataFrame
# from pyspark.sql.functions import lit

# countries = ['SU', 'OD', 'ET', 'ER', 'DJ', 'SO', 'UG', 'KE']

# dfs = []
# for co in countries:
#     c = co.lower()
#     cdf = spark.sql(f"SELECT * FROM news_media.horn_africa_gdelt_gsgembed_{c}_2w_slv")
#     cdf = cdf.withColumn('COUNTRY', lit(co))
#     dfs.append(cdf)

# dfs = reduce(DataFrame.unionAll, dfs)
# cols = ['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'] + list(np.arange(512).astype(str))
# dfs = dfs.select(*cols)
# dfs.write.mode('append').format('delta').saveAsTable(f'news_media.horn_africa_gdelt_gsgembed_2w_a1_5050_slv')

# COMMAND ----------


