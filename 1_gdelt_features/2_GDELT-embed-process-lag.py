# Databricks notebook source
# MAGIC %md
# MAGIC **What**: This notebook creates lagged embeddings variables ready for modeling (or PCA). It must be run after the 
# MAGIC `1_GDELT-embed-process-rollup` notbook. Designate the number of lags through the `N_LAGS` variable.
# MAGIC
# MAGIC **How**: Set the variables in `util/db_table.py`. Dates should already be set for the Events dataset download.  
# MAGIC   
# MAGIC **Note**: The `START_DATE` must be adjusted forward by the number of lags in order for it to have lags (this is done in the code). This notebook does not take long to run and does not require a Job cluster.

# COMMAND ----------

import datetime as dt

from pyspark.sql import functions as F
from pyspark.sql import Window

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, GDELT_EMBED_PROCESS_TABLE, GDELT_EMBED_PROCESS_LAG_TABLE, N_LAGS

# COMMAND ----------

# sanity check in job run
print(START_DATE, '-', END_DATE)
print(N_LAGS)

# COMMAND ----------

# IMPORTANT: START_DATE must be adjust forward by the number of lags for the START_DATE itself to have to have lags
START_DATE = (dt.datetime.strptime(START_DATE, '%Y-%m-%d') - dt.timedelta(weeks=N_LAGS)).date()

# filter to date range needed
df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EMBED_PROCESS_TABLE}")
print(df.count())
df = df.filter((df['STARTDATE'] >= START_DATE) & (df['ENDDATE'] < dt.datetime.strptime(END_DATE, '%Y-%m-%d').date()))
print(df.count())

# COMMAND ----------

cols = [str(x) for x in range(512)]
all_lags = df.select("STARTDATE", "ENDDATE", "ADMIN1", "COUNTRY")

keep_cols = []
for offset in range(1, N_LAGS+1):
    window = Window.partitionBy(F.col("COUNTRY"), F.col("ADMIN1")).orderBy(F.col("STARTDATE"))
    keep_cols.extend([f"{c}_t-{offset}" for c in cols])
    df_lag = df.select(
        "STARTDATE", "ADMIN1", "COUNTRY",
        *[F.lag(c, offset=offset).over(window).alias(f"{c}_t-{offset}") for c in cols]
    )
    all_lags = all_lags.alias('a').join(df_lag.alias('d'), (F.col('a.STARTDATE')==F.col('d.STARTDATE')) & (F.col('a.ADMIN1')==F.col('d.ADMIN1')) & (F.col('a.COUNTRY')==F.col('d.COUNTRY')), how='left')
    all_lags = all_lags.select('a.STARTDATE', 'a.ENDDATE', 'a.ADMIN1', 'a.COUNTRY', *keep_cols)

# this drops cases without enough lags
all_lags = all_lags.na.drop()

# COMMAND ----------

all_lags.write.mode('append').format('delta').saveAsTable(f"{DATABASE_NAME}.{GDELT_EMBED_PROCESS_LAG_TABLE}")

# COMMAND ----------


