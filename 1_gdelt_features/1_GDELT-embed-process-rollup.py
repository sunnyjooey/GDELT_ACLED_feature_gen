# Databricks notebook source
# MAGIC %md
# MAGIC **What**: This notebook creates an averaged embeddings dataset ready for modeling (or lagging or PCA). It is dependent on and must be run after the `0_GDELT-event-process` notebook. Designate the time period for averaging in number of weeks and the weight to be given the admin 1 level data (the countrywide data will be weighted 1 - admin%)
# MAGIC
# MAGIC **How**: Set the variables in `util/db_table.py`. Dates should already be set for the Events dataset download.  
# MAGIC   
# MAGIC **Note**: Run this notebook as a Job as it takes a lot of compute resources to run.

# COMMAND ----------

import numpy as np
import pandas as pd
import datetime as dt
import functools

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, GDELT_EMBED_TABLE, GDELT_EMBED_PROCESS_TABLE, GDELT_EVENT_PROCESS_TABLE, N_WEEK, COUNTRY_CODES

# COMMAND ----------

# weight of admin1 data, weight of CO (national) data is 1 - adm_pct
adm_pct = 0.8
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
    # read in events data 
    evtslv = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EVENT_PROCESS_TABLE} WHERE COUNTRY=='{CO}'")
    evtslv = evtslv.filter((evtslv['DATEADDED'] >= dt.datetime.strptime(START_DATE, '%Y-%m-%d').date()) & (evtslv['DATEADDED'] < dt.datetime.strptime(END_DATE, '%Y-%m-%d').date()))
    # merge events and embeddings
    co = evtslv.join(emb, evtslv.SOURCEURL==emb.url, how='left')
    cols = ['DATEADDED', 'ADMIN1', 'COUNTRY'] + list(np.arange(512).astype(str))
    co = co.select(*cols)

    # groupby n week intervals
    co = co.groupBy(F.window(F.col("DATEADDED"), n_week, "1 week", "-3 day"), 'ADMIN1', 'COUNTRY').mean()

    # parce out start and end time
    co = co.withColumn('STARTDATE', F.to_date(co['window']['start']))
    co = co.withColumn('ENDDATE', F.to_date(co['window']['end']))
    emb_cols = [f'avg({i})' for i in np.arange(512)]
    cols = ['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'] + emb_cols
    co = co.select(*cols)

    # split CO and admin data
    co_df = co.filter(co.ADMIN1==CO).select('STARTDATE', 'ENDDATE', 'ADMIN1', *emb_cols)
    co_df = co_df.toDF(*[f'{c}_' for c in co_df.columns])
    adm_df = co.filter(co.ADMIN1!=CO).select('STARTDATE', 'ADMIN1', *emb_cols)
    # merge to create all combos of date and admin 1
    co_df = co_df.withColumn('tmp_', F.lit(1))
    adm = adm_df.select('ADMIN1').distinct()
    adm = adm.withColumn('tmp', F.lit(1))
    co_df = co_df.join(adm, co_df.tmp_==adm.tmp).drop('ADMIN1_', 'tmp_', 'tmp')

    # CO data and admin data together into wiiiide table
    m = co_df.alias('c').join(adm_df.alias('a'), (F.col('c.STARTDATE_')==F.col('a.STARTDATE')) & (F.col('c.ADMIN1')==F.col('a.ADMIN1')), how='outer')
    # if admin is missing, fill in with CO data
    for col in emb_cols:
        m = m.withColumn(col, F.coalesce(col, f"{col}_"))
    
    # weighted average between admin and CO data
    if adm_pct is not None:
        co_pct = 1 - adm_pct
        # cycle through admin data cols
        for col in emb_cols:
            # CO data col
            co_col = f'{col}_'
            # calculate weighted average
            udfco = F.udf(lambda co: co * co_pct, DoubleType())
            m = m.withColumn(co_col, udfco(m[co_col]))
            udfadm = F.udf(lambda adm: adm * adm_pct, DoubleType())
            m = m.withColumn(col, udfadm(m[col]))
            udffin = F.udf(lambda co, adm: co + adm, DoubleType())
            m = m.withColumn(col, udffin(m[co_col], m[col]))
    
    # cleaning
    m = m.withColumn('COUNTRY', F.lit(CO))
    m = m.select('STARTDATE_', 'ENDDATE_', 'c.ADMIN1', 'COUNTRY', *emb_cols)
    m = m.toDF('STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY', *list(np.arange(512).astype(str)))

    # save
    m.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, GDELT_EMBED_PROCESS_TABLE))
    print(CO, 'done')

# COMMAND ----------


