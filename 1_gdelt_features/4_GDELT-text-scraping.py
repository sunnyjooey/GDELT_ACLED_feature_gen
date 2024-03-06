# Databricks notebook source
import numpy as np
import pandas as pd
import datetime as dt

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, GDELT_EVENT_PROCESS_TABLE, GDELT_SCRAPED_TEXT_TABLE

# COMMAND ----------

# sanity check in job run
print(START_DATE, '-', END_DATE)

# COMMAND ----------

# readin embed data
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EVENT_PROCESS_TABLE}")
print(emb.count())
# there are many duplicates in the embeddings data - keep only the first occurrence by url
emb = emb.orderBy('DATEADDED').coalesce(1).dropDuplicates(subset = ['SOURCEURL'])
print(emb.count())

# COMMAND ----------

# filter to date range needed
emb = emb.filter((emb['DATEADDED'] >= dt.datetime.strptime(START_DATE, '%Y-%m-%d').date()) & (emb['DATEADDED'] < dt.datetime.strptime(END_DATE, '%Y-%m-%d').date()))
print(emb.count())

# COMMAND ----------

# MAGIC %md
# MAGIC Note: keep all columns (date, url, admin1, country) and adda column for the scraped text

# COMMAND ----------

# save
# df.write.mode('append').format('delta').saveAsTable(f"{DATABASE_NAME}.{GDELT_SCRAPED_TEXT_TABLE}")
