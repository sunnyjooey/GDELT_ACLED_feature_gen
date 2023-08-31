# Databricks notebook source
import numpy as np
import pandas as pd
import datetime as dt

from pyspark.sql.functions import date_format, col
from functools import reduce

# COMMAND ----------

DATABASE_NAME = 'news_media'

# CHANGE ME!!
MODEL_TABLE_NAME = 'horn_africa_model_escbin_emb_confhist_lagpca_m51_gld'

EMB_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_pca_120_per_t_slv'
ACLED_TABLE_NAME = 'horn_africa_acled_sumfat_1w_slv'
GEO_STATIC_TABLE_NAME = 'horn_africa_geo_popdens2020_static_slv'
TREND_STATIC_TABLE_NAME = 'horn_africa_acled_conftrend_static_ct1_slv'
OUTCOME_TABLE_NAME = 'horn_africa_acled_outcome_fatal_escbin_1w_pct_v2_slv'

# COMMAND ----------

out = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{OUTCOME_TABLE_NAME}")
conf = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_TABLE_NAME}")
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EMB_TABLE_NAME}")
geo = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GEO_STATIC_TABLE_NAME}")
trend = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{TREND_STATIC_TABLE_NAME}")

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


