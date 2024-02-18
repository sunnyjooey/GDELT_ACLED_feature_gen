# Databricks notebook source
# MAGIC %md
# MAGIC Notebook for generating ACLED conflict history features with 2 week intervals and 1 week sliding windows  
# MAGIC Includes both `sum-deaths` and `time-since` features  
# MAGIC ***WARNING***: The start and end dates for querying ACLED data need to be manually calculated! It has not been functionalized!

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

from db_table import DATABASE_NAME, ACLED_CONFL_HIST_2_TABLE, COUNTRY_KEYS
from util import get_all_acled, get_one_co_data

# COMMAND ----------

# 2-week intervals starting on monday
INTERVAL = '2W-MON'

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

lag = pd.DataFrame()

for CO, CO_ACLED_NO in COUNTRY_KEYS.items():
    # query data to one country
    df = get_one_co_data(df_all, CO_ACLED_NO, 'ACLED_Admin1')

    # data_start_date: where to start the data (beginning of the lags), 
    # # # calculate by multiplying num_lags and INTERVAL and subtracting from where to start the feature set 
    # # # feature set start date is: 2019, 12, 30 - make sure this and data_start_date are in line with INTERVAL (a MONDAY!)
    # data_end_date: where to cut off the feature set
    d1 = make_lagged_features(df, 3, 'TimeFK_Event_Date', INTERVAL, dt.datetime(2019, 11, 18), dt.datetime(2023, 5, 1), 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')
    # doesn't seem to be a way to implement sliding window, so doing this manually
    # feature start date is 2020, 1, 6
    d2 = make_lagged_features(df, 3, 'TimeFK_Event_Date', INTERVAL, dt.datetime(2019, 11, 25), dt.datetime(2023, 5, 1), 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')

    # concat together, sort, clean
    d = pd.concat([d1, d2])
    d = d.sort_values('TimeFK_Event_Date')
    d['COUNTRY'] = CO
    lag = pd.concat([lag, d])

lag = lag.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'ACLED_Admin1':'ADMIN1'})

# COMMAND ----------

# DBTITLE 1,Create Time Units Since Features
def get_time_since(col, m_since_num):
    boolean_array = np.asarray(col >= m_since_num)
    if boolean_array.sum() == 0:
        time_since = np.array([len(boolean_array)] * len(boolean_array))
    else:
        nz = np.nonzero(boolean_array)
        mat = np.arange(len(boolean_array)).reshape(-1, 1) - nz
        mat[mat < 1] = len(boolean_array)
        time_since = np.min(mat, axis=1)
    return time_since


def get_time_since_df(df, m_since_lst, date_col, freq, admin_col, event_type_col, value_col, start_time, end_time):
    # query data
    df = df.loc[df[date_col] >= start_time, :]
    adm = list(df[admin_col].unique())
    evt = list(df[event_type_col].unique())
    # create intervals
    idx = pd.interval_range(start=start_time, end=end_time, freq=freq, closed='left')
    df[date_col] = idx[idx.get_indexer(df[date_col])]
    df[date_col] = df[date_col].apply(lambda x: x.left.date())
    # create wide pivot table 
    piv = pd.DataFrame(df.groupby([date_col, admin_col, event_type_col])[value_col].sum()).unstack().fillna(0)
    piv.columns = list(piv.columns.droplevel(0))
    piv = piv.reset_index()
    # create key table with all date and admin1s
    idx = pd.DataFrame([i.left.date() for i in idx], columns=[date_col])
    admins = pd.DataFrame(adm, columns=[admin_col])
    idx['tmp'] = 1
    admins['tmp'] = 1
    key = pd.merge(idx, admins, on=['tmp'])
    key = key.drop('tmp', axis=1)
    # merge together - this is to ensure all time intervals and admins are covered!
    data = pd.merge(key, piv, left_on=[date_col, admin_col], right_on=[date_col, admin_col], how='left')
    data[evt] = data[evt].fillna(0)

    # get time since (ex. at least 5 deaths due to protests)
    cols = pd.MultiIndex.from_product([adm, evt])
    date_index = idx[date_col]
    time_since_data = pd.DataFrame(index=date_index)
    for m_num in m_since_lst:
        for col in cols:
            colname = (col[0], col[1] + f'_since_{m_num}_death')  
            sub = data.loc[data[admin_col]==col[0], [date_col, col[1]]]
            time_since_data[colname] = get_time_since(sub[col[1]], m_num) 

    # filter to after 2020
    after_start = date_index[date_index >= dt.date(2019, 12, 30)]
    time_since_data = time_since_data.loc[after_start, : ]
    time_since_data.columns = pd.MultiIndex.from_tuples(time_since_data.columns)
    # stack the data (time and admin1 are indices)
    admin_levs = df[admin_col].cat.categories
    stacked_data = pd.DataFrame()
    for admin in admin_levs:
        adm = time_since_data[admin]
        adm.index = pd.MultiIndex.from_product([adm.index, [admin]])
        stacked_data = pd.concat([stacked_data, adm], axis=0)
    stacked_data = stacked_data.sort_index()
    return stacked_data

# COMMAND ----------

ts = pd.DataFrame()

for CO, CO_ACLED_NO in COUNTRY_KEY.items():
    # query data
    df = get_one_co_data(df_all, CO_ACLED_NO, 'ACLED_Admin1')

    # convert admin to category - make sure admins are not left out in groupby
    s1 = get_time_since_df(df, [1, 5, 20], 'TimeFK_Event_Date', INTERVAL, 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', dt.datetime(2011,1,1,0,0,0), dt.datetime(2023,5,1,0,0,0))
    s2 = get_time_since_df(df, [1, 5, 20], 'TimeFK_Event_Date', INTERVAL, 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', dt.datetime(2011,1,8,0,0,0), dt.datetime(2023,5,1,0,0,0))
    # make sure max is the same
    assert s1.max().max() == s2.max().max(), 'adjust start_time or end_time to make sure maxes match!'

    # concat together, sort, clean
    s = pd.concat([s1, s2])
    s = s.reset_index().sort_values('TimeFK_Event_Date')
    s['COUNTRY'] = CO
    ts = pd.concat([ts, s])

ts = ts.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'level_1':'ADMIN1'})

# COMMAND ----------

# keep only common startdates
lag['STARTDATE'] = lag['STARTDATE'].astype(str)
ts['STARTDATE'] = ts['STARTDATE'].astype(str)
times = set(lag['STARTDATE']).intersection(set(ts['STARTDATE']))
lag = lag[lag['STARTDATE'].isin(times)]
ts = ts[ts['STARTDATE'].isin(times)]

# COMMAND ----------

# merge features together
mrg = pd.merge(lag, ts, left_on=['STARTDATE','ADMIN1','COUNTRY'], right_on=['STARTDATE','ADMIN1','COUNTRY'], how='outer')

# COMMAND ----------

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

mrg.write.mode('append').format('delta').saveAsTable(f"{DATABASE_NAME}.{ACLED_CONFL_HIST_2_TABLE}")

# COMMAND ----------


