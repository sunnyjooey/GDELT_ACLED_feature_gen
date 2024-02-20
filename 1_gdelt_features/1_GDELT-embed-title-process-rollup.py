# Databricks notebook source
import numpy as np
import pandas as pd
import datetime as dt

import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, GDELT_EMBED_TABLE, GDELT_TITLE_FILL_TABLE, GDELT_TITLE_CONCAT_TABLE, N_WEEK, COUNTRY_CODES

# COMMAND ----------

# period of time for averaging 
n_week = f"{N_WEEK} week"

# IMPORTANT - rollups are from Monday - Sunday
# for best results, START_DATE and END_DATE should both be a Monday (weekday = 0)

# COMMAND ----------

# readin embed data
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EMBED_TABLE}")
print(emb.count())
# there are many duplicates in the embeddings data - keep only the first occurrence by url
emb = emb.orderBy('DATEADDED').coalesce(1).dropDuplicates(subset = ['url'])
print(emb.count())

# COMMAND ----------

# filter to date range needed
emb = emb.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
emb = emb.withColumn('DATEADDED', F.to_date('DATEADDED'))
emb = emb.filter((emb['DATEADDED'] >= dt.datetime.strptime(START_DATE, '%Y-%m-%d').date()) & (emb['DATEADDED'] < dt.datetime.strptime(END_DATE, '%Y-%m-%d').date()))
emb = emb.drop('DATEADDED')
print(emb.count())

# COMMAND ----------

# do one country at a time
for CO in COUNTRY_CODES:
    # read in events data and filter to date range
    evtslv = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EVENT_PROCESS_TABLE} WHERE COUNTRY=='{CO}'")
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
    m = m.fillna('', subset=['TITLE_', 'TITLE'])
    concat_txt_udf = udf(lambda t_, t: t_ if t=='' else t + ' [SEP] ' + t_, StringType())
    m = m.withColumn('TITLE', concat_txt_udf(m.TITLE_, m.TITLE))

    # cleaning
    m_fill = m_fill.withColumn('COUNTRY', F.lit(CO))
    m_fill = m_fill.select('STARTDATE_', 'ENDDATE_', 'c.ADMIN1', 'COUNTRY', 'TITLE')
    m_fill = m_fill.toDF('STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY', 'TITLE')

    m = m.withColumn('COUNTRY', F.lit(CO))
    m = m.select('STARTDATE_', 'ENDDATE_', 'c.ADMIN1', 'COUNTRY', 'TITLE')
    m = m.toDF('STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY', 'TITLE')

    # save
    m_fill.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, GDELT_TITLE_FILL_TABLE))
    m.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, GDELT_TITLE_CONCAT_TABLE))
    print(CO, 'done')

# COMMAND ----------


