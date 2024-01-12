# Databricks notebook source
# MAGIC %md
# MAGIC Notebook for lagging embed variables

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

DATABASE_NAME = 'news_media'
# CHANGE ME!!
INPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_slv'
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_gld'

# COMMAND ----------

df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")

# COMMAND ----------

num_lags = 4
cols = [str(x) for x in range(512)]

# COMMAND ----------

all_lags = df.select("STARTDATE", "ENDDATE", "ADMIN1", "COUNTRY")
keep_cols = []
for offset in range(1, num_lags+1):
    window = Window.partitionBy(F.col("COUNTRY"), F.col("ADMIN1")).orderBy(F.col("STARTDATE"))
    keep_cols.extend([f"{c}_t-{offset}" for c in cols])
    df_lag = df.select(
        "STARTDATE", "ADMIN1", "COUNTRY",
        *[F.lag(c, offset=offset).over(window).alias(f"{c}_t-{offset}") for c in cols]
    )
    all_lags = all_lags.alias('a').join(df_lag.alias('d'), (F.col('a.STARTDATE')==F.col('d.STARTDATE')) & (F.col('a.ADMIN1')==F.col('d.ADMIN1')) & (F.col('a.COUNTRY')==F.col('d.COUNTRY')), how='left')
    all_lags = all_lags.select('a.STARTDATE', 'a.ENDDATE', 'a.ADMIN1', 'a.COUNTRY', *keep_cols)

all_lags = all_lags.na.drop()

# COMMAND ----------

display(all_lags)

# COMMAND ----------

all_lags.write.mode('append').format('delta').saveAsTable(f"{DATABASE_NAME}.{OUTPUT_TABLE_NAME}")

# COMMAND ----------


