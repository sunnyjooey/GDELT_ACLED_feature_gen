# Databricks notebook source
# MAGIC %md
# MAGIC ## Dependent variable construction 

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

# import ACLED data function
df_all = get_all_acled()

# COMMAND ----------

# filter to HoA countries and start/end dates
cnty_codes = [v for v in COUNTRY_KEYS.values()]
start_date = dt.datetime.strptime(START_DATE, '%Y-%m-%d')
end_date = dt.datetime.strptime(END_DATE, '%Y-%m-%d') 
df = get_cnty_date_data(df_all, cnty_codes, start_date, end_date, 'ACLED_Admin1', 'TimeFK_Event_Date')

# COMMAND ----------

"""
IMPORTANT GROUPING NOTE:
    closed='left' means it includes the left/start date and excludes the right/end date
    closed='right' means it includes the right/end date and excludes the left/start date 
    label='left' means the start date is the label
    example: closed='left', label='left' means a label of 2019-12-30 includes it and up to 2020-01-05 (excludes 2020-01-06)
"""

def group_process_data(data):
    data = data.copy()
    # Create admin1_country_dict
    admin1_country_dict = df.set_index('ACLED_Admin1')['CountryFK'].to_dict()

    # Perform grouping and aggregation
    data = df.groupby([pd.Grouper(key='TimeFK_Event_Date', freq='W-MON', closed='left', label='left'), 'ACLED_Admin1']).agg({'ACLED_Fatalities': 'sum', 'CountryFK': 'first'}).reset_index()

    # Map admin1_country_dict to create 'CountryFK' column
    data['CountryFK'] = data['ACLED_Admin1'].map(admin1_country_dict)

    # change column names for easier merging and cleaning
    data = data.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'ACLED_Admin1':'ADMIN1', 'ACLED_Fatalities':'FATALSUM', 'CountryFK':'COUNTRY'})

    # change to 2 letter codes for merging
    country_keys = {val:key for key, val in COUNTRY_KEYS.items()}
    data['COUNTRY'] = data['COUNTRY'].map(country_keys)

    # drop no-admin1 rows
    data = data[data['ADMIN1']!='']
    
    # fill in known missing CO info
    data.loc[data['ADMIN1']=='Rwampara', 'COUNTRY'] = 'UG' 

    return data


# COMMAND ----------

data = group_process_data(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1.2 Make sure dependent and independent variables have the same admin1s

# COMMAND ----------

### this is a manual check!
# get all admin1, country from feature dataset to make sure none are missing
adm1 = spark.sql(f'SELECT * FROM {DATABASE_NAME}.{GDELT_TITLE_FILL_TABLE}')
adm1 = adm1.select('ADMIN1', 'COUNTRY').distinct().rdd.map(lambda r: (r[0], r[1])).collect()

data['set'] = data.apply(lambda row: (row['ADMIN1'], row['COUNTRY']), axis=1)
print('Total number of admin1s:', len(adm1))

print("In gdelt data but not in acled data - may have no fatalities")
print([x for x in adm1 if x not in list(data['set'].unique())])

# Note: 'Bahr el Ghazal' and 'Equatoria' are no longer valid
print("In acled data but not in gdelt data - note some admins are no longer valid")
print([x for x in list(data['set'].unique()) if x not in adm1])

# COMMAND ----------

def put_admin(adm_lst, put_in=[], take_out=[]):
    # take out of admin list
    adm_lst = [tup for tup in adm_lst if tup not in take_out]
    # put into admin list
    adm_lst.extend(put_in)
    print('Total number of admin1s:', len(adm_lst))
    return adm_lst

# Note: input admin1 in gdelt data but missing in acled (these have no fatalities)
# take out admin1 in gdelt data but missing in acled that have NO ADMIN INFO
# Note: admin1 in acled but not in gdelt will be dropped in the merge, no need to do anything
adm1 = put_admin(adm1, put_in=[('Busia', 'KE'), ('Arta', 'DJ')], take_out=[('SNNP', 'ET'), (None, 'OD'), (None, 'ER')])

# COMMAND ----------

# make sure each admin1 has all the time intervals - achieve this by merges
d = pd.DataFrame(list(product(*[data.STARTDATE.unique(), adm1])), columns=['STARTDATE','ADMIN1_CO'])
d['ADMIN1'] = d.apply(lambda row: row['ADMIN1_CO'][0], axis=1)
d['COUNTRY'] = d.apply(lambda row: row['ADMIN1_CO'][1], axis=1)
d.drop('ADMIN1_CO', axis=1, inplace=True)

data.drop('set', axis=1, inplace=True)
data = pd.merge(d, data, how='left')

# COMMAND ----------

# check nans - nans should only exist for admin1s added manually above
print(data.isnull().sum())
data[pd.isnull(data).any(axis=1)].drop_duplicates(['ADMIN1','COUNTRY'])

# COMMAND ----------

# fill in missing fatalsum with 0 - these are places with no acled fatalities
data = data.fillna({'FATALSUM': 0})
# these are erroneous - drop from data
data = data.dropna(subset=['COUNTRY'])
# final check
print(data.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Create escalation variable

# COMMAND ----------

def get_escalation_binary(data, fat_column, admin1_col):
    # Create a copy of the DataFrame to prevent modifications to the original object
    data = data.copy()
    # Calculate increase in fatalities
    data['abs_change'] = data.groupby(admin1_col)[fat_column].diff()
    # Percentage increase
    smoothing_factor = 1e-10
    data['pct_increase'] = (data['abs_change'] / (data.groupby(admin1_col)[fat_column].shift() + smoothing_factor)) * 100

    return data


def create_bin_esc_pct(data, pct, abs=None):
    data = data.copy()

    # Create a new column for binary_escalation
    if abs == None:
        outcome = f'bin_esc_{pct}'
    else:
        outcome = f'bin_esc_{abs}_{pct}'
    data[outcome] = 0

    if abs == None:
        # Apply thresholds for binary_escalation
        data.loc[data['pct_increase'] >= pct, outcome] = 1
    else:
        data.loc[(data['abs_change'] >= abs) & (data['pct_increase'] >= pct), outcome] = 1

    # check
    print('Number of escalation points:', len(data[data[outcome]==1]))
    return data

# COMMAND ----------

data = get_escalation_binary(data, 'FATALSUM', 'ADMIN1')
data = data.dropna() # this drops first time interval

# COMMAND ----------

# create bin esc variables
data = create_bin_esc_pct(data, 30)
data = create_bin_esc_pct(data, 50)
data = create_bin_esc_pct(data, 100)

data = create_bin_esc_pct(data, 30, 5)
data = create_bin_esc_pct(data, 50, 5)
data = create_bin_esc_pct(data, 100, 5)

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


