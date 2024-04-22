# Databricks notebook source
# MAGIC %md
# MAGIC **What**: This notebook creates an averaged embeddings dataset ready for modeling (or lagging or PCA). It is dependent on and must be run after the `0_GDELT-event-process` notebook. Designate the time period for averaging in number of weeks and the weight to be given the admin 1 level data (the countrywide data will be weighted 1 - admin%)
# MAGIC
# MAGIC **How**: Set the variables in `util/db_table.py`. Dates should already be set for the Events dataset download.  
# MAGIC   
# MAGIC **Note**: It takes a 126 GB 36 core Job cluster about 1 hour to run, seemingly regardless of the number of months, but it is recommended to run in several month or yearly segments in case of errors.

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

# sanity check in job run
print(START_DATE, '-', END_DATE)
print(N_WEEK)

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

emb_num = 512
# do one country at a time
for CO in COUNTRY_CODES:
    # read in events data 
    evtslv = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EVENT_PROCESS_TABLE} WHERE COUNTRY=='{CO}'")
    evtslv = evtslv.filter((evtslv['DATEADDED'] >= dt.datetime.strptime(START_DATE, '%Y-%m-%d').date()) & (evtslv['DATEADDED'] < dt.datetime.strptime(END_DATE, '%Y-%m-%d').date()))
    # merge events and embeddings to get needed embeddings
    co = evtslv.join(emb, evtslv.SOURCEURL==emb.url, how='left')
    cols = ['DATEADDED', 'ADMIN1', 'COUNTRY'] + list(np.arange(512).astype(str))
    co = co.select(*cols)

    # groupby n week intervals
    co = co.groupBy(F.window(F.col("DATEADDED"), n_week, "1 week", "-3 day"), 'ADMIN1', 'COUNTRY').mean()

    # parse out start and end time from windows and rename columns
    co = co.withColumn('STARTDATE', F.to_date(co['window']['start']))
    co = co.withColumn('ENDDATE', F.to_date(co['window']['end']))
    emb_cols = [f'avg({i})' for i in np.arange(emb_num)]
    cols = ['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'] + emb_cols
    co = co.select(*cols)

    # split CO and admin data
    co_df = co.filter(co.ADMIN1==CO).select('STARTDATE', 'ENDDATE', 'ADMIN1', *emb_cols)
    adm_df = co.filter(co.ADMIN1!=CO).select('STARTDATE', 'ENDDATE', 'ADMIN1', *emb_cols)

    ### account for missing data at CO and admin1 levels ###
    ### and proportionally combine the two ###
    # merge to create all combos of admin 1
    co_mini = adm_df.select('STARTDATE', 'ADMIN1')
    adm_mini = co_mini.select('ADMIN1').distinct().toDF('ADMIN1_')
    adm_mini = adm_mini.withColumn('tmp', F.lit(1))
    co_mini = co_mini.withColumn('tmp_', F.lit(1))
    combo_master = co_mini.join(adm_mini, co_mini.tmp_==adm_mini.tmp).drop('ADMIN1', 'tmp_', 'tmp')
    combo_master = combo_master.toDF('STARTDATE_', 'ADMIN1_')

    # by startdate, which admin1 and CO has data
    co_mini = co_mini.drop('tmp_')
    keep_master = combo_master.join(co_mini, (combo_master.STARTDATE_==co_mini.STARTDATE) & (combo_master.ADMIN1_==co_mini.ADMIN1), how='outer')
    co_mini = co_df.select('STARTDATE').toDF('STARTDATE__')
    keep_master = co_mini.join(keep_master, co_mini.STARTDATE__==keep_master.STARTDATE_, how='outer')
    keep_master = keep_master.distinct()

    # startdates with CO and admin1 data - m1
    co_adm = keep_master.filter((keep_master.STARTDATE__.isNotNull()) & (keep_master.STARTDATE.isNotNull()))
    # startdates with only CO and no admin1 data - m2
    co_only = keep_master.filter((keep_master.STARTDATE__.isNotNull()) & (keep_master.STARTDATE.isNull()))
    # startdates with no CO and only admin1 data - m3
    adm_only = keep_master.filter((keep_master.STARTDATE__.isNull()) & (keep_master.STARTDATE.isNotNull()))

    ## startdates with no CO and only admin1 data - adm_only
    m3 = adm_only.select('STARTDATE','ADMIN1').join(adm_df, on=['STARTDATE','ADMIN1'], how='left').select('STARTDATE','ENDDATE','ADMIN1',*emb_cols)
    m3 = m3.toDF('STARTDATE','ENDDATE','ADMIN1',*list(np.arange(emb_num).astype(str)))

    ## startdates with only CO and no admin1 data - co_only
    co_only = co_only.select('STARTDATE_', 'ADMIN1_')
    m2 = co_only.join(co_df, co_only.STARTDATE_==co_df.STARTDATE).select('STARTDATE','ENDDATE','ADMIN1_', *emb_cols)
    m2 = m2.toDF('STARTDATE','ENDDATE','ADMIN1',*list(np.arange(emb_num).astype(str)))

    ## startdates with CO and admin1 data - co_adm
    # multiply embeddings by proportion
    co_pct = 1 - adm_pct
    co_df_mult = co_df.withColumn("arr", F.struct(*[(F.col(x) * co_pct).alias(x) for x in emb_cols])).select("STARTDATE", "ENDDATE", "ADMIN1", "arr.*")
    adm_df_mult = adm_df.withColumn("arr", F.struct(*[(F.col(x) * adm_pct).alias(x) for x in emb_cols])).select("STARTDATE", "ENDDATE", "ADMIN1", "arr.*")

    # join with df that has startdates with CO and admin1 data
    adm_df_mult_mg = co_adm.select('STARTDATE','ADMIN1').join(adm_df_mult, on=['STARTDATE','ADMIN1'], how='left')

    co = co_df_mult.select('STARTDATE', 'ADMIN1')
    adm = adm_df_mult_mg.select('STARTDATE', 'ADMIN1')

    mg = adm.alias('a').join(co.alias('c'), on=['STARTDATE'])
    mg = mg.select('STARTDATE','a.ADMIN1','c.ADMIN1').toDF('STARTDATE_','ADMIN1_', 'ADMIN1_c')

    co_df_mult_mg = mg.join(co_df_mult, (mg.ADMIN1_c==co_df_mult.ADMIN1) & (mg.STARTDATE_==co_df_mult.STARTDATE), how='left')
    co_df_mult_mg = co_df_mult_mg.select('STARTDATE_', 'ADMIN1_', 'ENDDATE', *emb_cols).toDF('STARTDATE', 'ADMIN1', 'ENDDATE', *emb_cols)

    m1 = adm_df_mult_mg.union(co_df_mult_mg)
    m1 = m1.groupby(['STARTDATE','ENDDATE','ADMIN1']).agg(*[F.sum(c) for c in emb_cols])
    m1 = m1.toDF('STARTDATE', 'ENDDATE', 'ADMIN1', *list(np.arange(emb_num).astype(str)))

    # concat together
    m = m1.union(m2).union(m3)
    m = m.orderBy('STARTDATE', 'ADMIN1')

    # save
    m.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, GDELT_EMBED_PROCESS_TABLE))
    print(CO, 'done')

# COMMAND ----------


