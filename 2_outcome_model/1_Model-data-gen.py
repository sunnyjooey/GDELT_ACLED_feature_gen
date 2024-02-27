# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is not yet re-organized!

# COMMAND ----------

# import libraries
import numpy as np
import pandas as pd
import datetime as dt
from functools import reduce

from pyspark.sql.functions import date_format, col

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, MODEL_TABLE_NAME, GDELT_TITLE_FILL_TABLE, ACLED_CONFL_HIST_1_TABLE, ACLED_CONFL_TREND_TABLE, GEO_POP_DENSE_AGESEX_TABLE, ACLED_OUTCOME_TABLE

# COMMAND ----------

# read-in data
out = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_OUTCOME_TABLE}")
conf = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_CONFL_HIST_1_TABLE}")
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_TITLE_FILL_TABLE}")
geo = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GEO_POP_DENSE_AGESEX_TABLE}")
trend = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_CONFL_TREND_TABLE}")

# COMMAND ----------

out = out.withColumn("STARTDATE", date_format(col("STARTDATE"), "yyyy-MM-dd"))
conf = conf.withColumn("STARTDATE", date_format(col("STARTDATE"), "yyyy-MM-dd"))
emb = emb.withColumn("ENDDATE", date_format(col("ENDDATE"), "yyyy-MM-dd"))
print(out.count())
print(conf.count())
print(emb.count())

# COMMAND ----------

display(out)

# COMMAND ----------

display(conf)

# COMMAND ----------

display(emb)

# COMMAND ----------

# join conflict history and embedding features
m1 = conf.join(emb, (conf.STARTDATE==emb.STARTDATE) & (conf.ADMIN1==emb.ADMIN1) & (conf.COUNTRY==emb.COUNTRY)).drop(emb.STARTDATE).drop(emb.ENDDATE).drop(emb.ADMIN1).drop(emb.COUNTRY)
print(m1.count())

# COMMAND ----------

# join in static geo features
m2 = m1.join(geo, ['COUNTRY', 'ADMIN1'], 'left').drop(geo.ADMIN1).drop(geo.COUNTRY)
print(m2.count())

# COMMAND ----------

# join in static conflict trend dummy features
trend = trend.select('COUNTRY', 'ADMIN1', 'conflict_trend_1', 'conflict_trend_2')
m2 = m2.join(trend, ['COUNTRY', 'ADMIN1'], 'left').drop(trend.ADMIN1).drop(trend.COUNTRY)
print(m2.count())

# COMMAND ----------

# make sure no null values result from the merge
cols = [col(c) for c in m2.columns]
filter_expr = reduce(lambda a, b: a | b.isNull(), cols[1:], cols[0].isNull())
assert m2.filter(filter_expr).count() != 0, 'YOU HAVE NULL VALUES IN YOUR DATA!'

# COMMAND ----------

# join in outcome data
m3 = m2.join(out, ['COUNTRY', 'ADMIN1', 'STARTDATE']).drop(out.STARTDATE).drop(out.ADMIN1).drop(out.COUNTRY)
m3 = m3.orderBy('STARTDATE')
print(m3.count())

# COMMAND ----------

display(m3)

# COMMAND ----------

# save
m3.write.mode('append').format('delta').saveAsTable(f'{DATABASE_NAME}.{MODEL_TABLE_NAME}')

# COMMAND ----------


