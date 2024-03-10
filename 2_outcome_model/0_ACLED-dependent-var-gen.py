# Databricks notebook source
# MAGIC %md
# MAGIC ### Dependent variable construction 
# MAGIC **What**: This notebook generates a set of outcome variables (escalation binaries, fatalities sum) from the ACLED data.  
# MAGIC   
# MAGIC **How**: Set the variables in `util/db_table.py`. Dates should already be set for the Events dataset download.  
# MAGIC   
# MAGIC **Note**: This notebook assumes a table of all CO, admin1 pairs exists in `util/hoa_co_admin.csv`. This is to ensure all admin1s appear even if there were no fatalities in the week.

# COMMAND ----------

# import libraries
import pandas as pd
import numpy as np
import datetime as dt
from  itertools import product

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, GDELT_TITLE_FILL_TABLE, ACLED_OUTCOME_TABLE, COUNTRY_KEYS
from util import get_all_acled, get_cnty_date_data

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.1 Dependent variable cleaning and prep

# COMMAND ----------

# full list of CO and admin1 pairs
co_admin = pd.read_csv('../util/hoa_co_admin.csv')

# COMMAND ----------

# import ACLED data function
df_all = get_all_acled()

# COMMAND ----------

# filter ACLED to HoA countries and start/end dates
cnty_codes = [v for v in COUNTRY_KEYS.values()]
start_date = dt.datetime.strptime(START_DATE, '%Y-%m-%d')
end_date = dt.datetime.strptime(END_DATE, '%Y-%m-%d') 
df = get_cnty_date_data(df_all, cnty_codes, start_date, end_date, 'ACLED_Admin1', 'TimeFK_Event_Date')

# COMMAND ----------

"""
Sum up ACLED fatalities by week (starting Monday) and admin1
IMPORTANT GROUPING NOTE:
    closed='left' means it includes the left/start date and excludes the right/end date
    closed='right' means it includes the right/end date and excludes the left/start date 
    label='left' means the start date is the label
    example: closed='left', label='left' means a label of 2019-12-30 includes it and up to 2020-01-05 (excludes 2020-01-06)
"""

def group_process_data(data):
    data = data.copy()
    # change to 2 letter codes for merging
    country_keys = {val:key for key, val in COUNTRY_KEYS.items()}
    data['CountryFK'] = data['CountryFK'].map(country_keys)
    # create CO, admin1 pairs for aggregation
    data['CO_Admin_Set'] = data.apply(lambda row: (row['CountryFK'], row['ACLED_Admin1']), axis=1)

    ### perform grouping and aggregation ###
    data = data.groupby([pd.Grouper(key='TimeFK_Event_Date', freq='W-MON', closed='left', label='left'), 'CO_Admin_Set']).agg({'ACLED_Fatalities': 'sum'}).reset_index()

    # split back to CO and admin1
    data['COUNTRY'] = data.apply(lambda row: row['CO_Admin_Set'][0], axis=1)
    data['ADMIN1'] = data.apply(lambda row: row['CO_Admin_Set'][1], axis=1)
    # change column names for easier merging and cleaning
    data = data.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'ACLED_Fatalities':'FATALSUM'})
    # drop no-admin1 rows
    data = data[data['ADMIN1']!='']
    # fill in known missing CO info
    data.loc[data['ADMIN1']=='Rwampara', 'COUNTRY'] = 'UG' 
    # reorder columns
    data = data.loc[:, ['STARTDATE', 'COUNTRY', 'ADMIN1', 'FATALSUM']]

    return data


# COMMAND ----------

# Sum up ACLED fatalities by week (starting Monday) and admin1
data = group_process_data(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2 Make sure dependent and independent variables have the same admin1s

# COMMAND ----------

# make sure each admin1 has all the time intervals - achieve this by merges
adm1 = list(co_admin.apply(lambda row: (row['COUNTRY'], row['ADMIN1']), axis=1))
d = pd.DataFrame(list(product(*[data.STARTDATE.unique(), adm1])), columns=['STARTDATE','ADMIN1_CO'])
d['ADMIN1'] = d.apply(lambda row: row['ADMIN1_CO'][1], axis=1)
d['COUNTRY'] = d.apply(lambda row: row['ADMIN1_CO'][0], axis=1)
d.drop('ADMIN1_CO', axis=1, inplace=True)

# merge (date, admin1, CO) sets with fatality sum data
da = pd.merge(d, data, how='left')
# fill na with 0 since there were no fatalities in those weeks
da['FATALSUM'] = da['FATALSUM'].fillna(0)
# change to date object
da['STARTDATE'] = da['STARTDATE'].apply(lambda x: x.date())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Create escalation variable

# COMMAND ----------

def get_pct_increase(data, fat_column, admin1_col):
    # Create a copy of the DataFrame to prevent modifications to the original object
    data = data.copy()
    # Calculate increase in fatalities
    data['abs_change'] = data.groupby(admin1_col)[fat_column].diff()
    # Percentage increase
    smoothing_factor = 1e-10
    data['pct_increase'] = (data['abs_change'] / (data.groupby(admin1_col)[fat_column].shift() + smoothing_factor)) * 100
    return data


def create_escalation_binary(data, pct, abs=None):
    data = data.copy()

    # Create a new column for binary_escalation
    if abs == None:
        outcome = f'bin_esc_{pct}'
    else:
        outcome = f'bin_esc_{abs}_{pct}'
    data[outcome] = 0

    # Apply thresholds for binary_escalation
    if abs == None:
        data.loc[data['pct_increase'] >= pct, outcome] = 1
        print(f'Escalation threshold: {pct}% increase; Number of escalation points:', len(data[data[outcome]==1]))
    else:
        data.loc[(data['abs_change'] >= abs) & (data['pct_increase'] >= pct), outcome] = 1
        print(f'Escalation threshold: over {abs} and {pct}% increase; Number of escalation points:', len(data[data[outcome]==1]))
    return data

# COMMAND ----------

# step 1 - get percent increase
data = get_pct_increase(da, 'FATALSUM', 'ADMIN1')
# this drops first time interval
data = data.dropna() 

# COMMAND ----------

# create bin esc variables
data = create_escalation_binary(data, 30)
data = create_escalation_binary(data, 50)
data = create_escalation_binary(data, 100)

data = create_escalation_binary(data, 30, 5)
data = create_escalation_binary(data, 50, 5)
data = create_escalation_binary(data, 100, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Save it to Pyspark 

# COMMAND ----------

# convert to spark dataframe
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
data = spark.createDataFrame(data)

# COMMAND ----------

# save in delta lake
data.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ACLED_OUTCOME_TABLE))

# COMMAND ----------


