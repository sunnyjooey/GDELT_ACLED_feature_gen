# Databricks notebook source
import pandas as pd
import numpy as np
import datetime as dt

# COMMAND ----------

# country code
CO_ACLED_NO = 214
# 2-week intervals starting on monday
INTERVAL = '2W-MON'

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

database_host = dbutils.secrets.get(scope='warehouse_scope', key='database_host')
database_port = dbutils.secrets.get(scope='warehouse_scope', key='database_port')
user = dbutils.secrets.get(scope='warehouse_scope', key='user')
password = dbutils.secrets.get(scope='warehouse_scope', key='password')

database_name = "UNDP_DW_CRD"
table = "dbo.CRD_ACLED"
url = f"jdbc:sqlserver://{database_host}:{database_port};databaseName={database_name};"

df_all = (spark.read
      .format("com.microsoft.sqlserver.jdbc.spark")
      .option("url", url)
      .option("dbtable", table)
      .option("user", user)
      .option("password", password)
      .load()
    )

# COMMAND ----------

def get_data(df, cnty_code, admin_col):
    # sudan country code - filter first before converting to pandas
    df = df.filter(df.CountryFK==cnty_code)
    df = df.toPandas()
    # convert admin to category - make sure admins are not left out in groupby
    df[admin_col] = df[admin_col].astype('category')
    # create year-month column
    df['TimeFK_Event_Date'] = df['TimeFK_Event_Date'].apply(lambda x: dt.datetime.strptime(str(x),'%Y%m%d'))    
    return df

# COMMAND ----------

df = get_data(df_all, CO_ACLED_NO, 'ACLED_Admin1')

# COMMAND ----------

def make_lagged_features(df, num_lags, date_col, freq, data_start_date, index_start_date, admin_col, event_type_col, value_col, agg_func):
    df = df.loc[df[date_col] >= data_start_date, :]
    # create wide pivot table 
    # include the starting date, have that as label
    piv = pd.DataFrame(df.groupby([pd.Grouper(key=date_col, freq=freq, closed='left', label='left'), admin_col, event_type_col])[value_col].agg(agg_func)).unstack().fillna(0)
    # number of names in admin level
    num_adm = len(df[admin_col].cat.categories)
    # keep track of columns
    orig_cols = list(piv.columns.droplevel(0))
    cols = orig_cols.copy()
    piv.columns = cols

    # create lags
    for i in range(1, num_lags+1):
        cols_tmin = [f'{col}_t-{i}' for col in orig_cols]
        cols.extend(cols_tmin)
        piv.reindex(columns=cols)
        piv[cols_tmin] = piv[orig_cols].shift(num_adm * i).values

    # filter to after start date
    date_index = piv.index.levels[0] 
    after_start = date_index[date_index >= index_start_date]
    piv = piv.loc[after_start, : ]
    # drop non-lagged cols - uncomment to check work
    piv = piv.loc[:, [c for c in piv.columns if c not in orig_cols]]
    return piv

# COMMAND ----------

# data_start_date: where to start the data including the lags, calculate by multiplying num_lags and INTERVAL (starting point is index_start_date)
# index_start_date: where to start the feature set
# both dates should be a Monday (match with INTERVAL)
d1 = make_lagged_features(df, 3, 'TimeFK_Event_Date', INTERVAL, dt.datetime(2019, 11, 18, 0, 0, 0), dt.datetime(2019, 12, 30, 0, 0, 0), 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')
# couldn't figure how to do sliding window in pandas, so doing this manually
d2 = make_lagged_features(df, 3, 'TimeFK_Event_Date', INTERVAL, dt.datetime(2019, 11, 25, 0, 0, 0), dt.datetime(2020, 1, 6, 0, 0, 0), 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')

# COMMAND ----------

# concat together and sort
d = pd.concat([d1, d2])
d = d.reset_index()
d = d.sort_values('TimeFK_Event_Date')

# COMMAND ----------

# dd = df[(df['TimeFK_Event_Date']>=dt.datetime(2019,12,23,0,0,0)) & (df['TimeFK_Event_Date']<dt.datetime(2020,1,6,0,0,0)) & (df['ACLED_Event_Type']=='Battles') & (df['ACLED_Admin1']=='Red Sea')]['ACLED_Fatalities'].sum()
# dd

# COMMAND ----------



# COMMAND ----------

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


def get_time_since_df(df, m_since_lst, date_col, freq, admin_col, event_type_col, value_col, start_time, num_lags):
    # create wide pivot table
    piv = pd.DataFrame(df.groupby([pd.Grouper(key=date_col, freq=freq), admin_col, event_type_col])[value_col].sum()).unstack().fillna(0)
    # cols of (admin name, event type) tuples
    cols = pd.MultiIndex.from_product([df[admin_col].unique(), df[event_type_col].unique()])
    piv.columns = list(piv.columns.droplevel(0))

    # get time since (ex. at least 5 deaths due to protests)
    time_since_data = pd.DataFrame()
    idx = pd.IndexSlice
    for m_num in m_since_lst:
        for col in cols:
            colname = (col[0], col[1] + f'_since_{m_num}_death')  
            time_since_data[colname] = get_time_since(piv.loc[idx[:,col[0]], col[1]], 5)     

    # date index - reset
    date_index = piv.index.levels[0] 
    time_since_data.index = date_index
    after_start = date_index[date_index >= start_time]
    time_since_data = time_since_data.loc[after_start, : ]
    time_since_data.columns = pd.MultiIndex.from_tuples(time_since_data.columns)

    admin_levs = df[admin_col].cat.categories
    stacked_data = pd.DataFrame()
    for admin in admin_levs:
        adm = time_since_data[admin]
        adm.index = pd.MultiIndex.from_product([adm.index, [admin]])
        stacked_data = pd.concat([stacked_data, adm], axis=0)
    stacked_data = stacked_data.sort_index()

    return stacked_data

# COMMAND ----------

# convert admin to category - make sure admins are not left out in groupby
df['ACLED_Event_Type'] = df['ACLED_Event_Type'].astype('category')
s = get_time_since_df(df, [1, 5], 'TimeFK_Event_Date', INTERVAL, 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', dt.datetime(2020,1,1,0,0,0), 2)

# COMMAND ----------

# this means that on jan 2020,
# it was 3 months since at least 1 fatality due to battles in central darfur
# it was 6 months since at least 5 fatalities due to protests in north kordofan
s.head(20)

# COMMAND ----------

s.tail(20)

# COMMAND ----------


