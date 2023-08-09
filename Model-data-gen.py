# Databricks notebook source
import numpy as np
import pandas as pd
import datetime as dt

from pyspark.sql.functions import date_format, col

# COMMAND ----------

DATABASE_NAME = 'news_media'
# CHANGE ME!!
EMB_TABLE_NAME = 'horn_africa_gdelt_gsgembed_2w_a1_8020_slv'
ACLED_TABLE_NAME = 'horn_africa_acled_confhist_2w_slv'
OUTCOME_TABLE_NAME = 'horn_africa_acled_outcome_fatal_escbin_1w_pct_slv'
MODEL_TABLE_NAME = 'horn_africa_model_escbin_emb_confhist_m3_gld'

# COMMAND ----------

out = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{OUTCOME_TABLE_NAME}")
conf = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_TABLE_NAME}")
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EMB_TABLE_NAME}")

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

# match embedding endate with conflict history start date
m1 = conf.join(emb, (conf.STARTDATE==emb.ENDDATE) & (conf.ADMIN1==emb.ADMIN1)).drop(emb.STARTDATE).drop(emb.ENDDATE).drop(emb.ADMIN1).drop(emb.COUNTRY)
print(m1.count())

# COMMAND ----------

display(m1)

# COMMAND ----------

m2 = m1.join(out, (m1.STARTDATE==out.STARTDATE) & (m1.ADMIN1==out.ADMIN1)).drop(out.STARTDATE).drop(out.ADMIN1).drop(out.COUNTRY)

# COMMAND ----------

print(m2.count())
display(m2)

# COMMAND ----------

m2.write.mode('append').format('delta').saveAsTable(f'{DATABASE_NAME}.{MODEL_TABLE_NAME}')

# COMMAND ----------


