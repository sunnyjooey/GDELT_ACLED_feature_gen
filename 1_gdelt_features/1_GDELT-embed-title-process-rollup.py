# Databricks notebook source
import numpy as np
import pandas as pd
import datetime as dt

import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# COMMAND ----------

# IMPORTANT - rollups are from Monday - Sunday
# for best results, start_date and end_date should both be a Monday (weekday = 0)
start_date = '2019-12-30'  # inclusive
end_date = '2023-05-01'  # exclusive: download does not include this day 

# COMMAND ----------

# period of time for averaging 
n_week = "1 week"

# COMMAND ----------

DATABASE_NAME = 'news_media'
EVTSLV_TABLE_NAME = 'horn_africa_gdelt_events_cameo1_slv'
EMB_TABLE_NAME = 'horn_africa_gdelt_gsgembed_brz'
# CHANGE ME!!
OUTPUT_TABLE_NAME_FILL = 'horn_africa_gdelt_cameo1_gsgemb_title_fill_1w_slv'
OUTPUT_TABLE_NAME_CONCAT = 'horn_africa_gdelt_cameo1_gsgemb_title_concat_1w_slv'

# COMMAND ----------

# readin embed data
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EMB_TABLE_NAME}")
print(emb.count())
# there are many duplicates in the embeddings data - keep only the first occurrence by url
emb = emb.orderBy('DATEADDED').coalesce(1).dropDuplicates(subset = ['url'])
print(emb.count())

# COMMAND ----------

# filter to date range needed
emb = emb.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
emb = emb.withColumn('DATEADDED', F.to_date('DATEADDED'))
emb = emb.filter((emb['DATEADDED'] >= dt.datetime.strptime(start_date, '%Y-%m-%d').date()) & (emb['DATEADDED'] < dt.datetime.strptime(end_date, '%Y-%m-%d').date()))
emb = emb.drop('DATEADDED')
print(emb.count())

# COMMAND ----------

# do one country at a time
countries = ['SU', 'OD', 'ET', 'ER', 'DJ', 'SO', 'UG', 'KE']

for CO in countries:
    # read in events data and filter to date range
    evtslv = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVTSLV_TABLE_NAME} WHERE COUNTRY=='{CO}'")
    evtslv = evtslv.filter((evtslv['DATEADDED'] >= dt.datetime.strptime(start_date, '%Y-%m-%d').date()) & (evtslv['DATEADDED'] < dt.datetime.strptime(end_date, '%Y-%m-%d').date()))
    # merge events and embeddings
    co = evtslv.join(emb, evtslv.SOURCEURL==emb.url, how='left')
    cols = ['DATEADDED', 'ADMIN1', 'COUNTRY', 'title'] 
    co = co.select(*cols)
    # drop duplicate rows
    co = co.dropDuplicates()

    # groupby n week intervals
    co = co.groupBy(F.window(F.col("DATEADDED"), n_week, "1 week", "-3 day"), 'ADMIN1', 'COUNTRY').agg(F.concat_ws(' [SEP] ', F.collect_list('title')).alias('TITLE'))

    # parce out start and end time
    co = co.withColumn('STARTDATE', F.to_date(co['window']['start']))
    co = co.withColumn('ENDDATE', F.to_date(co['window']['end']))
    cols = ['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY', 'TITLE']
    co = co.select(*cols)

    # split CO and admin data
    co_df = co.filter(co.ADMIN1==CO).select('STARTDATE', 'ENDDATE', 'ADMIN1', 'TITLE')
    co_df = co_df.toDF(*[f'{c}_' for c in co_df.columns])
    adm_df = co.filter(co.ADMIN1!=CO).select('STARTDATE', 'ADMIN1', 'TITLE')
    # merge to create all combos of date and admin 1
    co_df = co_df.withColumn('tmp_', F.lit(1))
    adm = adm_df.select('ADMIN1').distinct()
    adm = adm.withColumn('tmp', F.lit(1))
    co_df = co_df.join(adm, co_df.tmp_==adm.tmp).drop('ADMIN1_', 'tmp_', 'tmp')

    # CO data and admin data together into wiiiide table
    m = co_df.alias('c').join(adm_df.alias('a'), (F.col('c.STARTDATE_')==F.col('a.STARTDATE')) & (F.col('c.ADMIN1')==F.col('a.ADMIN1')), how='outer')

    # 1. fill with CO-wide titles only if no titles for admin 
    m_fill = m.withColumn("TITLE", F.coalesce(m.TITLE, m.TITLE_)) 

    # 2. fill with CO-wide titles if no titles for admin, otherwise concat admin titles and CO-wide titles
    concat_txt_udf = udf(lambda t_, t: t_ if t is None else t + ' [SEP] ' + t_, StringType())
    m = m.withColumn('TITLE', concat_txt_udf(m.TITLE_, m.TITLE))

    # cleaning
    m_fill = m_fill.withColumn('COUNTRY', F.lit(CO))
    m_fill = m_fill.select('STARTDATE_', 'ENDDATE_', 'c.ADMIN1', 'COUNTRY', 'TITLE')
    m_fill = m_fill.toDF('STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY', 'TITLE')

    m = m.withColumn('COUNTRY', F.lit(CO))
    m = m.select('STARTDATE_', 'ENDDATE_', 'c.ADMIN1', 'COUNTRY', 'TITLE')
    m = m.toDF('STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY', 'TITLE')

    # save
    m_fill.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME_FILL))
    m.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME_CONCAT))
    print(CO, 'done')
