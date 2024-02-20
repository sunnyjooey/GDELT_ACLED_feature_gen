# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is meant to be used to experiment / do visual checks. It is NOT meant to be automated in a job. Save only the table meant to be used in the modeling.    
# MAGIC Future: create this feature on a rolling basis. Currently, there is some backward leakage because the static variable is based on trends during the whole outcome duration.

# COMMAND ----------

# import libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import DATABASE_NAME, ACLED_OUTCOME_TABLE, ACLED_CONFL_TREND_TABLE

# COMMAND ----------

DATABASE_NAME = 'news_media'
ACLED_OUTCOME_TABLE = 'horn_africa_acled_outcome_fatal_escbin_1w_pct_slv'
ACLED_CONFL_TREND_TABLE = 'horn_africa_acled_conftrend_static_ct1_slv'

# COMMAND ----------

# read in dependent variable
escalation_data = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{ACLED_OUTCOME_TABLE}")
escalation_data = escalation_data.toPandas()

# COMMAND ----------

# make data wide
e = escalation_data.loc[:, ['STARTDATE', 'ADMIN1', 'COUNTRY', 'FATALSUM']]
w = e.pivot(index=['COUNTRY','ADMIN1'], columns='STARTDATE', values='FATALSUM')

# COMMAND ----------

# create intermediary vars

# proportion of time intervals with at least x fatalities
def proportion_over_x(df, x):
    df[f'prop_{x}'] = df.apply(lambda row: len(row[row >= x]) / len(row), axis=1)
    return df

# 1 if the maximum fatality is over x, 0 otherwise
def max_over_x(df, x):
    df[f'max_{x}'] = df.apply(lambda row: 1 if max(row) >= x else 0, axis=1)
    return df

# number of time intervals with at least x fatalities
def num_over_x(df, x):
    df[f'n_{x}'] = df[df >= x].count(axis=1)
    return df

w = proportion_over_x(w, 5)
w = max_over_x(w, 100)
w = max_over_x(w, 50)
w = num_over_x(w, 100)

# COMMAND ----------

# create trend categories
def trend_cat(row):
    # high conflict
    if ((row['prop_5'] >= .2) and (row['max_100'] == 1)) or (row['n_100'] >=3):
        return 1
    # low conflict
    elif (row['prop_5'] < .1) and (row['max_50'] == 0): 
        return 3
    # medium conflict
    else:
        return 2

w['trend_cat1'] = w.apply(lambda row: trend_cat(row), axis=1)
w.groupby('trend_cat1').size()

# COMMAND ----------

# convert categories into dummies
dummies = pd.get_dummies(w['trend_cat1'], dtype=np.int64).rename(columns=lambda x: 'conflict_trend_' + str(x))
w = pd.concat([w, dummies], axis=1)
w = w.drop(['trend_cat1'], axis=1)

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC
# MAGIC # do a visual check
# MAGIC def graph_it(long_df, wide_df, filter_col, sample=None):
# MAGIC     cdf = pd.merge(long_df, wide_df.loc[:, filter_col], on='ADMIN1', how='left')
# MAGIC     cdf = cdf[cdf[filter_col] == 1]
# MAGIC     if sample != None:
# MAGIC         adm1s = random.sample(list(cdf.ADMIN1.unique()), sample)
# MAGIC         cdf = cdf[cdf.ADMIN1.isin(adm1s)]
# MAGIC     
# MAGIC     # Plotting the fatalities over time by country
# MAGIC     for adm1 in cdf.ADMIN1.unique():
# MAGIC         plt.figure(figsize=(30, 10))
# MAGIC         adm1_data = cdf[cdf['ADMIN1'] == adm1]
# MAGIC         adm1_data = adm1_data.sort_values('STARTDATE')
# MAGIC         plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'])
# MAGIC         plt.ylim(0, 500)
# MAGIC         
# MAGIC         plt.xlabel('Time')
# MAGIC         plt.xticks(rotation=90)
# MAGIC         plt.ylabel('Fatalities')
# MAGIC         plt.title(adm1)
# MAGIC         plt.show()

# COMMAND ----------

graph_it(escalation_data, w, 'conflict_trend_1')

# COMMAND ----------

graph_it(escalation_data, w, 'conflict_trend_2')

# COMMAND ----------

# too many in this category - sample only 20
graph_it(escalation_data, w, 'conflict_trend_3', 20)

# COMMAND ----------

# keep only some columns
w = w[['prop_5','max_100','max_50','n_100', 'conflict_trend_1','conflict_trend_2']]
w = w.reset_index()

# COMMAND ----------

# convert to spark dataframe
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
w = spark.createDataFrame(w)

# COMMAND ----------

# save in delta lake
# this will write if the table does not exist, but throw an error if it does exist
w.write.mode('errorifexists').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ACLED_CONFL_TREND_TABLE))
