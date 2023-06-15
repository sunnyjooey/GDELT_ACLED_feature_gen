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

escalation_data = spark.sql("SELECT * FROM news_media.horn_africa_acled_fatal_1w_10a10p_bin_gld")

# COMMAND ----------

escalation_data = escalation_data.toPandas()

# COMMAND ----------

escalation_data

# COMMAND ----------

escalation_data.COUNTRY.unique()

# COMMAND ----------

cdf = escalation_data[escalation_data['COUNTRY'] == 'KE']
plt.figure(figsize=(30, 10)) 

# Plotting the fatalities over time by country
for adm1 in cdf.ADMIN1.unique():
    adm1_data = cdf[cdf['ADMIN1'] == adm1]
    plt.plot(adm1_data['STARTDATE'], adm1_data['FATALSUM'], label=adm1)
    
escalation_df = cdf[cdf['binary_escalation'] == 1]
plt.plot(escalation_df['STARTDATE'], escalation_df['FATALSUM'], marker='o', linestyle='', color='black', label='Escalation')
plt.xlabel('Time')
plt.xticks(rotation=90)
plt.ylabel('Fatalities')
plt.title('Fatalities Over Time by Country')
plt.legend()
plt.show()

# COMMAND ----------


