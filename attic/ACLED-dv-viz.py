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

escalation_data[(escalation_data['ADMIN1']=='Blue Nile') & (escalation_data['binary_escalation_30']==1)]

# COMMAND ----------

escalation_data.COUNTRY.unique()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'UG']
ad = cdf['ADMIN1'].unique()[:70]
cdf = cdf[cdf['ADMIN1'].isin(ad)]
for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        
    escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.legend()
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'UG']
ad = cdf['ADMIN1'].unique()[70:]
cdf = cdf[cdf['ADMIN1'].isin(ad)]
for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        
    escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.legend()
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'KE']
ad = cdf['ADMIN1'].unique()
for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        
    escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.legend()
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'SO']
ad = cdf['ADMIN1'].unique()
for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        
    escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.legend()
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'ET']
ad = cdf['ADMIN1'].unique()
for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        
    escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.legend()
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'OD']
ad = cdf['ADMIN1'].unique()
for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        
    escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.legend()
    plt.show()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'SU']
ad = cdf['ADMIN1'].unique()
for a in ad:
    plt.figure(figsize=(30, 10)) 

    # Plotting the fatalities over time by country
    for adm1 in cdf.ADMIN1.unique():
        adm1_data = cdf[cdf['ADMIN1'] == adm1]
        adm1_data = adm1_data[adm1_data['ADMIN1']==a]
        adm1_data = adm1_data.sort_values('STARTDATE')
        plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
        
    escalation_df = cdf[cdf['binary_escalation_30'] == 1]
    plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.ylabel('Fatalities')
    plt.title('Fatalities Over Time by Country')
    plt.legend()
    plt.show()

# COMMAND ----------

news = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_events_a1_slv")

# COMMAND ----------

import datetime as dt

# COMMAND ----------

n1 = news.filter((news.ADMIN1=='Blue Nile') & (news.DATEADDED < dt.date(2022, 10, 24)) & (news.DATEADDED > dt.date(2022, 10, 2)))
n1 = n1.dropDuplicates(['SOURCEURL'])

# COMMAND ----------

display(n1)

# COMMAND ----------

def _nchars_leq_ntokens_approx(maxTokens):
    """
    Returns a number of characters very likely to correspond <= maxTokens
    """
    sqrt_margin = 0.5
    lin_margin = 1.010175047 #= e - 1.001 - sqrt(1 - sqrt_margin) #ensures return 1 when maxTokens=1
    return max( 0, int(maxTokens*math.exp(1) - lin_margin - math.sqrt(max(0,maxTokens - sqrt_margin) ) ))


# COMMAND ----------

import math
_nchars_leq_ntokens_approx(200)

# COMMAND ----------


