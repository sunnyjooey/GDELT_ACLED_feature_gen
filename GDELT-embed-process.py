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

# COMMAND ----------

co = CO.lower()
DATABASE_NAME = 'news_media'
EVTSLV_TABLE_NAME = f'horn_africa_gdelt_events_{co}_slv'
EMB_TABLE_NAME = 'horn_africa_gdelt_gsgembed_brz'
OUTPUT_TABLE_NAME = f'horn_africa_gdelt_gsgembed_{co}_2w_slv'

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
m = all_df.groupBy(F.window(F.col("DATEADDED"), "2 week"), 'ADMIN1').mean()
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
df1 = functools.reduce(DataFrame.union, collect_dfs)
df1 = df1.withColumn('STARTDATE', F.to_date(df1['DATE']['start']))
df1 = df1.withColumn('ENDDATE', F.to_date(df1['DATE']['end']))
cols = ['STARTDATE', 'ENDDATE', 'ADMIN1'] + list(np.arange(512).astype(str))
df1 = df1.select(*cols)
df1.count()

# COMMAND ----------

# off-set data by 1 week
all_df = all_df.filter(all_df['DATEADDED'] >= dt.date(2020, 1, 8))

# get average embeddings by time intervals (2-week) and admin 1
m = all_df.groupBy(F.window(F.col("DATEADDED"), "2 week"), 'ADMIN1').mean()
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
df2 = functools.reduce(DataFrame.union, collect_dfs)
df2 = df2.withColumn('STARTDATE', F.to_date(df2['DATE']['start']))
df2 = df2.withColumn('ENDDATE', F.to_date(df2['DATE']['end']))
cols = ['STARTDATE', 'ENDDATE', 'ADMIN1'] + list(np.arange(512).astype(str))
df2 = df2.select(*cols)
df2.count()

# COMMAND ----------

# concat together
df = df1.union(df2)
df.count()

# COMMAND ----------

# save
df.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))

# COMMAND ----------


