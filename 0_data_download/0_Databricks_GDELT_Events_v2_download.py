# Databricks notebook source
# MAGIC %md
# MAGIC This notebook downloads the GDELT Events (v2) datasets. Change `start_date`, `end_date`, and `country_codes`.

# COMMAND ----------

# import libraries
import math
import re
import os
import pandas as pd
import numpy as np
import datetime as dt
import urllib
import zipfile
import gc
from io import BytesIO
import warnings
warnings.simplefilter('ignore', FutureWarning)

from pyspark.sql.types import StructType, StructField, StringType

# COMMAND ----------

# Input Params

start_date = '2023-01-01'  # inclusive
end_date = '2023-05-01'  # exclusive: download does not include this day 
country_codes = ['SU', 'OD', 'ET', 'ER', 'DJ', 'SO', 'UG', 'KE']  # list of countries to keep
DATABASE_NAME = 'news_media'
TABLE_NAME = 'horn_africa_gdelt_events_brz'
ERROR_TABLE_NAME = 'horn_africa_errors'

# COMMAND ----------

# define search range (15 min chunks in each day)
def get_date_time_intervals(_start_date, _end_date):
    _date_time_range = []
    _range = pd.date_range(start=_start_date, end=_end_date, freq='15T')
    for _date in _range:
        _date_time_range.append(str(_date))
    return _date_time_range

# get dates
date_range = get_date_time_intervals(start_date, end_date)

# exclude last time frame published at midnight for last 15 min from day before
date_range = date_range[:-1]
print('Number of time intervals:', len(date_range))

# COMMAND ----------

# define GDELT schema
gdelt_event_url = "https://raw.githubusercontent.com/linwoodc3/gdelt2HeaderRows/master/schema_csvs/GDELT_2.0_Events_Column_Labels_Header_Row_Sep2016.csv"
gdelt_events_schema = pd.read_csv(gdelt_event_url, sep=',')
gdelt_events_headers = gdelt_events_schema['tableId'].values

# COMMAND ----------

# get data from Events 2.0 in batches

idx_date = 0
batch_size_date = 96
total_range = len(date_range)
num_batches_date = math.ceil(total_range / batch_size_date)
gdelt_data_full_search = pd.DataFrame()
error_df = pd.DataFrame()

for batch in range(num_batches_date):
    print('Batch:', batch)
    # instatiate df for batch of news
    _gdelt_data_batch = pd.DataFrame()

    for count, date in enumerate(date_range[idx_date:idx_date+batch_size_date]):
        if count % 48 == 0:
            print(date)
        _date = re.sub(r'[^\d+]','', date)
        try:
            # fetch gdelt events data
            _gdelt_data = pd.read_csv(f"http://data.gdeltproject.org/gdeltv2/{_date}.export.CSV.zip", 
                                      lineterminator='\n', delimiter='\t', encoding='utf8', header=None, compression='zip')
            _gdelt_data.columns = gdelt_events_headers
            # remove undated rows
            _gdelt_data = _gdelt_data.dropna(subset=['SQLDATE']) 
            # append news to df for batch
            _gdelt_data_batch = _gdelt_data_batch.append(_gdelt_data)
        except Exception as e:
            edf = pd.DataFrame({'date':[_date], 'data':['events2'], 'error':[str(e)]})
            error_df = error_df.append(edf)
            print(f'#### FAILED AT {date} ####')

    # reset index
    _gdelt_data_batch.reset_index(inplace=True, drop=True)
    # shape of unmodified events from batch
    print('ALL GDELT 2.0 events in batch:', _gdelt_data_batch.shape)
    # select data from events for defined country 
    # note: we are not filtering by Actor1CountryCode (and 2) because they do not seem to be accurate
    _gdelt_data_batch = _gdelt_data_batch.loc[(_gdelt_data_batch.ActionGeo_CountryCode.isin(country_codes)) | 
                                              (_gdelt_data_batch.Actor1Geo_CountryCode.isin(country_codes)) | 
                                              (_gdelt_data_batch.Actor2Geo_CountryCode.isin(country_codes))].copy()
    print('Number of country relevant events:', _gdelt_data_batch.shape[0])
    # reset index
    _gdelt_data_batch.reset_index(inplace=True, drop=True)
    # append batch to full search results df
    gdelt_data_full_search = gdelt_data_full_search.append(_gdelt_data_batch)
    # unpdate indices for next batch
    idx_date += batch_size_date

    # clear batch variables from memory
    del _gdelt_data_batch
    gc.collect()
    del _gdelt_data
    gc.collect()
    
gdelt_data_full_search.reset_index(inplace=True, drop=True)

# COMMAND ----------

# minor cleaning - drop columns
gdelt_data_full_search = gdelt_data_full_search.drop(columns=['MonthYear', 'Year', 'FractionDate'])

# COMMAND ----------

# define schema -- data is messy, so cast everything as string for now
schema = [StructField(col, StringType(), True) for col in gdelt_data_full_search.columns]

# COMMAND ----------

# convert to spark
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spdf = spark.createDataFrame(gdelt_data_full_search, StructType(schema))

# COMMAND ----------

# save output
spdf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, TABLE_NAME))

# COMMAND ----------

# save error table if any
if error_df.shape[0] > 0:
    eschema = [StructField(col, StringType(), True) for col in error_df.columns]
    spedf = spark.createDataFrame(error_df, StructType(eschema))
    spedf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, ERROR_TABLE_NAME))

# COMMAND ----------


