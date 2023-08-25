# Databricks notebook source
# MAGIC %md
# MAGIC ### Dependent variable construction 
# MAGIC
# MAGIC ### things to think about:

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# COMMAND ----------

escalation_data = spark.sql("SELECT * FROM news_media.horn_africa_acled_outcome_fatal_escbin_1w_pct_slv")

# COMMAND ----------

escalation_data = escalation_data.toPandas()

# COMMAND ----------

escalation_data

# COMMAND ----------

e = escalation_data[['STARTDATE' ,'ADMIN1',	'FATALSUM']]
w = e.pivot(index='ADMIN1', columns='STARTDATE', values='FATALSUM')

# COMMAND ----------

w['prop_0'] = w.apply(lambda x: len(x[x > 0]) / len(x), axis=1)

# COMMAND ----------

w['n_100'] = w[w >= 100].count(axis=1)

# COMMAND ----------

w[w >= 100].count(axis=1)

# COMMAND ----------

w['prop_5'] = w.apply(lambda x: len(x[x >= 5]) / len(x), axis=1)


# COMMAND ----------

w['max_50'] = w.apply(lambda x: 1 if max(x) >= 50 else 0, axis=1)

# COMMAND ----------

w['max_50'].sum()

# COMMAND ----------

def trend_cat(row):
    if ((row['prop_5'] >= .2) and (row['max_100'] == 1)) or (row['n_100'] >=3):
        return 1
    elif (row['prop_5'] < .2) and (row['max_50'] == 0):
        return 3
    else:
        return 2

# COMMAND ----------

w['trend_cat7'] = w.apply(lambda row: trend_cat(row), axis=1)

# COMMAND ----------

w.groupby('trend_cat7').size()

# COMMAND ----------

escalation_data = pd.merge(escalation_data, w.iloc[:, -1], on='ADMIN1', how='left')

# COMMAND ----------

cdf = escalation_data[escalation_data['trend_cat7']==1]
ad = cdf['ADMIN1'].unique()

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 500)
        
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['trend_cat7']==2]
ad = cdf['ADMIN1'].unique()

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 500)
        
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['trend_cat7']==3]
ad = cdf['ADMIN1'].unique()
ad = np.random.choice(ad, 20, replace=False)

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 100)
        
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------


