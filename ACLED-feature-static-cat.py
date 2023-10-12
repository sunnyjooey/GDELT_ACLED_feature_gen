# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

DATABASE_NAME = 'news_media'
INPUT_DATA_TABLE = 'horn_africa_acled_outcome_fatal_escbin_1w_pct_slv'
OUTPUT_DATA_TABLE = 'horn_africa_acled_conftrend_static_ct1_slv'

# COMMAND ----------

# read in dependent variable
escalation_data = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_DATA_TABLE}")
escalation_data = escalation_data.toPandas()

# COMMAND ----------

# make data wide
e = escalation_data[['STARTDATE' ,'ADMIN1',	'COUNTRY', 'FATALSUM']]
w = e.pivot(index=['COUNTRY','ADMIN1'], columns='STARTDATE', values='FATALSUM')

# COMMAND ----------

# create intermediary vars
def proportion_over_x(df, x):
    df[f'prop_{x}'] = df.apply(lambda row: len(row[row >= x]) / len(row), axis=1)
    return df

def max_over_x(df, x):
    df[f'max_{x}'] = df.apply(lambda row: 1 if max(row) >= x else 0, axis=1)
    return df

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

dummies = pd.get_dummies(w['trend_cat1']).rename(columns=lambda x: 'conflict_trend_' + str(x))
w = pd.concat([w, dummies], axis=1)
w = w.drop(['trend_cat1'], axis=1)

# COMMAND ----------

# import random
# %matplotlib inline

# def graph_it(long_df, wide_df, filter_col, sample=None):
#     cdf = pd.merge(long_df, wide_df.loc[:, filter_col], on='ADMIN1', how='left')
#     cdf = cdf[cdf[filter_col] == 1]
#     if sample != None:
#         adm1s = random.sample(list(cdf.ADMIN1.unique()), sample)
#         cdf = cdf[cdf.ADMIN1.isin(adm1s)]
    
#     # Plotting the fatalities over time by country
#     for adm1 in cdf.ADMIN1.unique():
#         plt.figure(figsize=(30, 10))
#         adm1_data = cdf[cdf['ADMIN1'] == adm1]
#         adm1_data = adm1_data.sort_values('STARTDATE')
#         plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'])
#         plt.ylim(0, 500)
        
#         plt.xlabel('Time')
#         plt.xticks(rotation=90)
#         plt.ylabel('Fatalities')
#         plt.title(adm1)
#         plt.show()

# graph_it(escalation_data, w, 'conflict_trend_1')

# COMMAND ----------

# graph_it(escalation_data, w, 'conflict_trend_2')

# COMMAND ----------

# graph_it(escalation_data, w, 'conflict_trend_3', 20)

# COMMAND ----------

w = w[['prop_5','max_100','max_50','n_100', 'conflict_trend_1','conflict_trend_2']]
w = w.reset_index()

# COMMAND ----------

# convert to spark dataframe
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
w = spark.createDataFrame(w)

# COMMAND ----------

# save in delta lake
w.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_DATA_TABLE))
