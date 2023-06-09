# Databricks notebook source
import pandas as pd
import numpy as np
import datetime as dt

# COMMAND ----------

# 2-week intervals starting on monday
INTERVAL = '2W-MON'

country_keys = {
    'SU': 214,
    'OD': 227,
    'ET': 108,
    'ER': 104,
    'DJ': 97,
    'SO': 224,
    'UG': 235,
    'KE': 175
}

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

# DBTITLE 1,Create Lagged Features
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

lag = pd.DataFrame()

for CO, CO_ACLED_NO in country_keys.items():
    # query data
    df = get_data(df_all, CO_ACLED_NO, 'ACLED_Admin1')

    # data_start_date: where to start the data (beginning of the lags), calculate by multiplying num_lags and INTERVAL and subtracting from index_start_date
    # index_start_date: where to start the feature set
    # both dates should be a Monday (or match with INTERVAL)
    d1 = make_lagged_features(df, 3, 'TimeFK_Event_Date', INTERVAL, dt.datetime(2019, 11, 18, 0, 0, 0), dt.datetime(2019, 12, 30, 0, 0, 0), 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')
    # doesn't seem to be a way to implement sliding window with pd.Grouper, so doing this manually
    d2 = make_lagged_features(df, 3, 'TimeFK_Event_Date', INTERVAL, dt.datetime(2019, 11, 25, 0, 0, 0), dt.datetime(2020, 1, 6, 0, 0, 0), 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')

    # concat together, sort, clean
    d = pd.concat([d1, d2])
    d = d.reset_index().sort_values('TimeFK_Event_Date')
    d['COUNTRY'] = CO
    lag = pd.concat([lag, d])

lag = lag.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'ACLED_Admin1':'ADMIN1'})

# COMMAND ----------

lag

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
    # merge together
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

for CO, CO_ACLED_NO in country_keys.items():
    # query data
    df = get_data(df_all, CO_ACLED_NO, 'ACLED_Admin1')

    # convert admin to category - make sure admins are not left out in groupby
    s1 = get_time_since_df(df, [1, 5, 20], 'TimeFK_Event_Date', INTERVAL, 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', dt.datetime(2011,1,1,0,0,0), dt.datetime(2023,5,1,0,0,0))
    s2 = get_time_since_df(df, [1, 5, 20], 'TimeFK_Event_Date', INTERVAL, 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', dt.datetime(2011,1,8,0,0,0), dt.datetime(2023,5,1,0,0,0))

    # concat together, sort, clean
    s = pd.concat([s1, s2])
    s = s.reset_index().sort_values('TimeFK_Event_Date')
    s['COUNTRY'] = CO
    ts = pd.concat([ts, s])

ts = ts.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'level_1':'ADMIN1'})

# COMMAND ----------

ts

# COMMAND ----------



# COMMAND ----------

ts['a'] = ts['STARTDATE'].astype(str)+ts['ADMIN1']
lag['a'] = lag['STARTDATE'].astype(str)+lag['ADMIN1']
a = list(lag.a)
b = list(ts.a)
c=[x for x in a if x not in b]
d=[x for x in b if x not in a]

# COMMAND ----------


