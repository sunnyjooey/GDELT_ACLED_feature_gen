# Databricks notebook source
# MAGIC %md
# MAGIC ***README: Note on Dates***  
# MAGIC All start and end dates are set to a MONDAY.  
# MAGIC All start dates are inclusive, while end dates are exclusive. This also applies to datasets where end dates are implied and not explicitly stated.
# MAGIC 1. **Outcome data**
# MAGIC     * `STARTDATE` is the date column. Here, a STARTDATE of 2023-05-15 refers to the interval from 2023-05-15 (inclusive) to 2023-05-22 (exclusive, or up to 2023-05-21) 
# MAGIC     * `FATALSUM` is the sum of fatalities from 2023-05-15 (inclusive) to 2023-05-22 (exclusive)
# MAGIC     * `pct_increase` is the percent increase from the previous time interval (in this example, the previous time interval is 2023-05-08 inclusive to 2023-05-15 exclusive)
# MAGIC     * `bin_esc_30` is a 30% or more pct_increase from the previous time interval 
# MAGIC
# MAGIC 2. **Lagged data**
# MAGIC     * This can be lagged GDELT (averaged embeddings) or ACLED data
# MAGIC     * Lagged variables will be labeld `t-n`, like `t-1`, `t-2`, etc.
# MAGIC     * `STARTDATE` is the starting date of the *current* interval and `t-1` refers to the *previous* interval. For example, if a STARTDATE is 2023-05-15, the `t-1` variable refers to the value during 2023-05-08 (inclusive) to 2023-05-15 (exclusive)  
# MAGIC
# MAGIC 3. **Un-lagged data**
# MAGIC     * This can be averaged embeddings or concatenated titles data without any t-n variables
# MAGIC     * `STARTDATE` is inclusive, while `ENDDATE` is exclusive. A STARTDATE of 2023-05-15 (inclusive) will have an ENDDATE of 2023-05-22 (exclusive, or up to 2023-05-21)
# MAGIC
# MAGIC 4. **Static data**
# MAGIC     * This can be population density or conflict trend categorical data
# MAGIC     * There are no dates, so merges must be done on `COUNTRY` and `ADMIN1` only
# MAGIC   
# MAGIC   
# MAGIC ***How to Merge on Dates***  
# MAGIC 1. *Lagged* data and Outcome data: merge on `STARTDATE`  
# MAGIC 2. *Un-lagged* data and Outcome data: merge on un-lagged data's `ENDDATE` with outcome data's `STARTDATE`.
# MAGIC
# MAGIC See this diagram for explanation  
# MAGIC ![time_interval](../util/img/time_interval_diagram.png)

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
outcome = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_OUTCOME_TABLE}")
conflict_hist = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_CONFL_HIST_1_TABLE}")
title_text = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_TITLE_FILL_TABLE}")
pop_dense = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GEO_POP_DENSE_AGESEX_TABLE}")
trend_category = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_CONFL_TREND_TABLE}")

# COMMAND ----------

# make sure dates are uniform before joining
outcome = outcome.withColumn("STARTDATE", date_format(col("STARTDATE"), "yyyy-MM-dd"))
conflict_hist = conflict_hist.withColumn("STARTDATE", date_format(col("STARTDATE"), "yyyy-MM-dd"))
title_text = title_text.withColumn("ENDDATE", date_format(col("ENDDATE"), "yyyy-MM-dd"))
# count of rows
print(outcome.count())
print(conflict_hist.count())
print(title_text.count())

# COMMAND ----------

# drop some columns from static feature datasets
pop_dense_cols = ['COUNTRY','ADMIN1','female_0_14_2020','male_0_14_2020','female_15_64_2020','male_15_64_2020','female_65_plus_2020','male_65_plus_2020']
trend_category_cols = ['COUNTRY','ADMIN1','conflict_trend_1','conflict_trend_2']
pop_dense = pop_dense.select(*pop_dense_cols)
trend_category = trend_category.select(*trend_category_cols)
# count of rows
print(pop_dense.count())
print(trend_category.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Joins

# COMMAND ----------

# join conflict history and text features
m1 = conflict_hist.join(title_text, (conflict_hist.STARTDATE==title_text.STARTDATE) & (conflict_hist.ADMIN1==title_text.ADMIN1) & (conflict_hist.COUNTRY==title_text.COUNTRY)).drop(title_text.STARTDATE).drop(title_text.ENDDATE).drop(title_text.ADMIN1).drop(title_text.COUNTRY)
print(m1.count())

# COMMAND ----------

# join in static geo features
m2 = m1.join(pop_dense, ['COUNTRY', 'ADMIN1'], 'left')
print(m2.count())

# COMMAND ----------

# join in static conflict trend dummy features
m2 = m2.join(trend_category, ['COUNTRY', 'ADMIN1'], 'left')
print(m2.count())

# COMMAND ----------

# make sure no null values result from the merge
cols = [col(c) for c in m2.columns]
filter_expr = reduce(lambda a, b: a | b.isNull(), cols[1:], cols[0].isNull())
assert m2.filter(filter_expr).count() != 0, 'YOU HAVE NULL VALUES IN YOUR DATA!'

# COMMAND ----------

# join in outcome data
m3 = m2.join(outcome, ['COUNTRY', 'ADMIN1', 'STARTDATE']).orderBy('STARTDATE')
print(m3.count())

# COMMAND ----------

display(m3)

# COMMAND ----------

# save
m3.write.mode('append').format('delta').saveAsTable(f'{DATABASE_NAME}.{MODEL_TABLE_NAME}')
