# Databricks notebook source
# MAGIC %md
# MAGIC **What**: Notebook for generating ACLED lagged conflict history features at 1 week intervals. Includes only `sum-deaths` features.  
# MAGIC   
# MAGIC **How**: Set the variables in util/db_table.py. Dates should already be set for the Events dataset download.
# MAGIC   
# MAGIC **Note**: Within the notebook, it takes about 7 minutes to run on 9 weeks of data.
# MAGIC

# COMMAND ----------

# import libraries
import pandas as pd
import numpy as np
import datetime as dt
from functools import reduce

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, ACLED_CONFL_HIST_1_TABLE, COUNTRY_KEYS, N_LAGS
from util import get_all_acled, get_one_co_data

# COMMAND ----------

# 1-week intervals starting on monday
INTERVAL = '1W-MON'

# COMMAND ----------

# import ACLED data function
df_all = get_all_acled()

# COMMAND ----------

# DBTITLE 1,Create Lagged Features
def make_lagged_features(df, num_lags, date_col, freq, data_start_date, data_end_date, admin_col, event_type_col, value_col, agg_func):
    # calculate the data_start_date to the beginning of the lags!
    df = df.loc[df[date_col] >= data_start_date, :]
    adm = list(df[admin_col].unique())
    evt = list(df[event_type_col].unique())
    # create date intervals (add 1 week to data_end_date to capture to the end of the data)
    idx = pd.interval_range(start=data_start_date, end=data_end_date+dt.timedelta(weeks=1), freq=freq, closed='left')
    df[date_col] = idx[idx.get_indexer(df[date_col])]
    df[date_col] = df[date_col].apply(lambda x: x.left.date())
    # create wide pivot table of [fatalities by event type]
    piv = pd.DataFrame(df.groupby([date_col, admin_col, event_type_col])[value_col].agg(agg_func)).unstack().fillna(0)
    piv.columns = list(piv.columns.droplevel(0))    
    piv = piv.reset_index()
    # create key table with all date and admin1 combos
    idx = pd.DataFrame([i.left.date() for i in idx], columns=[date_col])
    admins = pd.DataFrame(adm, columns=[admin_col])
    idx['tmp'] = 1
    admins['tmp'] = 1
    key = pd.merge(idx, admins, on=['tmp'])
    key = key.drop('tmp', axis=1)
    # merge together - this is to ensure all time intervals and admins are covered!
    piv = pd.merge(key, piv, left_on=[date_col, admin_col], right_on=[date_col, admin_col], how='left')
    piv[evt] = piv[evt].fillna(0)
    
    # set up for shifting data
    idx = list(idx[date_col])
    idx.sort()
    evt_cols = [c for c in piv.columns if c not in [date_col, admin_col]]
    ret = pd.DataFrame()
    # start at the num_lags-th date to account for the lags
    for i in range(num_lags, len(idx)):
        lag_lst = []
        # get lagged data
        for l in range(1, num_lags+1): 
            cols_tmin = [f'{col}_t-{l}' for col in evt_cols]
            # print(idx[i], idx[i-l])
            one_lag = piv.loc[piv[date_col]==idx[i-l], :].drop(date_col, axis=1)
            one_lag.columns = [admin_col] + cols_tmin
            lag_lst.append(one_lag)
        # combine to one dataframe
        all_lags = reduce(lambda x, y: pd.merge(x, y, on = admin_col), lag_lst)
        # keep track of the date we lagged from
        all_lags[date_col] = idx[i]
        ret = pd.concat([ret, all_lags])
    # filter to just the data we need
    ret = ret.loc[ret[date_col] < data_end_date.date(), :]
    return ret


# COMMAND ----------

# # # calculate ACLED data start and end date to query
# data_start_date: where to start the data (beginning of the lags)
# # # if INTERVAL is 1 week, calculate by subtracting N_LAGS weeks from the start of the feature set (2019, 12, 30 in GDELT)
# data_end_date: where to cut off the feature set
nweeks = -1 * N_LAGS  # this needs to be changed if intervals and windows don't match (e.g. 2w interval, 1w sliding window)
data_start_date = dt.datetime.strptime(START_DATE, '%Y-%m-%d') + dt.timedelta(weeks=nweeks)
data_end_date = dt.datetime.strptime(END_DATE, '%Y-%m-%d')

# COMMAND ----------

lag = pd.DataFrame()

for CO, CO_ACLED_NO in COUNTRY_KEYS.items():
    # query data to one country
    df = get_one_co_data(df_all, CO_ACLED_NO, 'ACLED_Admin1', 'TimeFK_Event_Date')
    # get all lagged features
    d = make_lagged_features(df, N_LAGS, 'TimeFK_Event_Date', INTERVAL, data_start_date, data_end_date, 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')
    d = d.sort_values('TimeFK_Event_Date')
    d['COUNTRY'] = CO
    lag = pd.concat([lag, d])

lag = lag.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'ACLED_Admin1':'ADMIN1'})

# COMMAND ----------

lag['STARTDATE'] = lag['STARTDATE'].astype(str)
mrg = lag.copy()
# data cleaning - these admin 1 don't exist anymore
mrg = mrg.loc[~((mrg['COUNTRY']=='SU') & (mrg['ADMIN1'].isin(['Bahr el Ghazal', 'Equatoria', 'Upper Nile']))), :] 
# fill in NAs - extensive checking has been done to make sure this is valid
mrg = mrg.fillna(0)

# COMMAND ----------

# reorder
cols = [c for c in mrg.columns if c not in ['STARTDATE', 'COUNTRY', 'ADMIN1']]
cols = ['STARTDATE', 'COUNTRY', 'ADMIN1'] + cols
mrg = mrg[cols]
mrg.columns = [c.replace('/','_').replace(' ','_') for c in mrg.columns]

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
mrg = spark.createDataFrame(mrg)

# COMMAND ----------

mrg.write.mode('append').format('delta').saveAsTable(f"{DATABASE_NAME}.{ACLED_CONFL_HIST_1_TABLE}")

# COMMAND ----------


