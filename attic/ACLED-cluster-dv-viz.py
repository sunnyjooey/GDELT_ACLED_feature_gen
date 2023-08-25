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

!pip3 install pip install numba==0.53.0
!pip install tslearn

# COMMAND ----------

from tslearn.clustering import TimeSeriesKMeans

# COMMAND ----------

SEED = 451
k2 = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=100, random_state=SEED).fit(w.values)
k3 = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=100, random_state=SEED).fit(w.values)
k4 = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=100, random_state=SEED).fit(w.values)
k5 = TimeSeriesKMeans(n_clusters=5, metric="dtw", max_iter=100, random_state=SEED).fit(w.values)
k6 = TimeSeriesKMeans(n_clusters=6, metric="dtw", max_iter=100, random_state=SEED).fit(w.values)
k7 = TimeSeriesKMeans(n_clusters=7, metric="dtw", max_iter=100, random_state=SEED).fit(w.values)
k8 = TimeSeriesKMeans(n_clusters=8, metric="dtw", max_iter=100, random_state=SEED).fit(w.values)

# COMMAND ----------

w['K2'] = k2.labels_
w['K3'] = k3.labels_
w['K4'] = k4.labels_
w['K5'] = k5.labels_
w['K6'] = k6.labels_
w['K7'] = k7.labels_
w['K8'] = k8.labels_

# COMMAND ----------

escalation_data = pd.merge(escalation_data, w.iloc[:, -7:], on='ADMIN1', how='left')

# COMMAND ----------

# these groups did not really make sense
w.groupby('K8').size()

# COMMAND ----------

escalation_data[(escalation_data['FATALSUM']>=20) & (escalation_data['FATALSUM']<1000)]['FATALSUM'].hist(bins=5)

# COMMAND ----------

pd.set_option('display.max_rows', None)

# COMMAND ----------

escalation_data[escalation_data['FATALSUM']!=0].groupby(['FATALSUM']).size()

# COMMAND ----------

small = []
fifty = []
hundred = []
twoh = []
threeh = []
fourh = []
beyond = []

for adm1 in escalation_data.ADMIN1.unique():
    adm1_data = escalation_data[escalation_data['ADMIN1'] == adm1]
    mx = max(adm1_data.FATALSUM)
    if mx >= 300:
        threeh.append(adm1)
    elif mx >= 200:
        twoh.append(adm1)
    elif mx >= 100:
        hundred.append(adm1)
    elif mx >= 50:
        fifty.append(adm1)
    else:
        small.append(adm1)

# COMMAND ----------

cdf = escalation_data
ad = np.random.choice(small, 20, replace=False)

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 50)
        
    # escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    # plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------

cdf = escalation_data
ad = threeh

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 500)
        
    # escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    # plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------

cdf = escalation_data
ad = twoh

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 300)
        
    # escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    # plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------

cdf = escalation_data
ad = hundred

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 200)
        
    # escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    # plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------

cdf = escalation_data
ad = fifty

for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        plt.ylim(0, 100)
        
    # escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    # plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.show()

# COMMAND ----------


