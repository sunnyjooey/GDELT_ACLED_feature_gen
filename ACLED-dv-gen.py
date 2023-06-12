# Databricks notebook source
# MAGIC %md
# MAGIC ### Dependent variable construction 
# MAGIC
# MAGIC ### things to think about:

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime as dt

# COMMAND ----------

# load acled data

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

def get_data(df, admin_col, cnty_codes):
    # Filter by Horn of Africa countries
    # horn of africa list 
    df = df.filter(df.CountryFK.isin(cnty_codes))
    
    # Convert to pandas dataframe
    df = df.toPandas()
    
    # Convert admin to category
    df[admin_col] = df[admin_col].astype('category')
    
    # Create year-month column
    df['TimeFK_Event_Date'] = df['TimeFK_Event_Date'].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m%d'))

    # filter dates after dec 30, 2019
    df = df[df['TimeFK_Event_Date'] >= dt.datetime(2019, 12, 30, 0, 0, 0)]

    return df

# COMMAND ----------

cnty_codes = [214, 227, 108, 104, 97, 224, 235, 175]
df = get_data(df_all,'ACLED_Admin1',cnty_codes) #get all horn of africa countries, Sudan:214, South Sudan:227, Ethiopia:108, Eritrea:104, Djibouti:97,Somalia:224,Uganda:235,Kenya:175

# COMMAND ----------

# MAGIC %md
# MAGIC #### Dependent variable cleaning and prep

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Define functions: 

# COMMAND ----------

def process_data(data):
    data = data.copy()
    # Create admin1_country_dict
    admin1_country_dict = df.set_index('ACLED_Admin1')['CountryFK'].to_dict()

    # Perform grouping and aggregation
    data = df.groupby([pd.Grouper(key='TimeFK_Event_Date', freq='W-MON', closed='left', label='left'), 'ACLED_Admin1']).agg({'ACLED_Fatalities': 'sum', 'CountryFK': 'first'}).reset_index()

    # Map admin1_country_dict to create 'CountryFK' column
    data['CountryFK'] = data['ACLED_Admin1'].map(admin1_country_dict)

    return data

# COMMAND ----------

def get_escalation_binary(data, fat_column,admin1_col):
    # Create a copy of the DataFrame to prevent modifications to the original object
    data = data.copy()

    # Calculate increase in fatalities
    data['abs_change'] = data.groupby(admin1_col)[fat_column].diff()

    # Percentage increase
    smoothing_factor = 1e-10
    data['pct_increase'] = (data['abs_change'] / (data.groupby(admin1_col)[fat_column].shift() + smoothing_factor)) * 100

    # Create a new column for binary_escalation
    data['binary_escalation'] = 0

    # Apply thresholds for binary_escalation
    data.loc[(data[fat_column] >= 100) & (data['pct_increase'] >= 10), 'binary_escalation'] = 1
    data.loc[(data[fat_column] < 100) & (data['abs_change'] >= 10), 'binary_escalation'] = 1

    # Return the modified DataFrame
    return data

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Process data and plot

# COMMAND ----------

data = process_data(df)

# COMMAND ----------

# change column names for easier merging and cleaning
data = data.rename(columns={'TimeFK_Event_Date':'STARTDATE', 'ACLED_Admin1':'ADMIN1', 'ACLED_Fatalities':'FATALSUM', 'CountryFK':'COUNTRY'})

# change to 2 letter codes for merging
country_keys = {
    214: 'SU',
    227: 'OD',
    108: 'ET',
    104: 'ER',
    97: 'DJ',
    224: 'SO',
    235: 'UG',
    175: 'KE'
}

data['COUNTRY'] = data['COUNTRY'].map(country_keys)

# COMMAND ----------

# read in feature set for checking
feat = spark.sql("SELECT * FROM news_media.horn_africa_acled_confhist_2w_gld")
feat = feat.toPandas()
# keep only one feature column to make checking easier
feat = feat.iloc[:, :4]

# COMMAND ----------

# filter fatal data to match feature set's dates
data = data.loc[data['STARTDATE'] <= feat['STARTDATE'].max()]
# change to string to make merge easier
data['STARTDATE'] = data['STARTDATE'].astype(str)  

# COMMAND ----------

# merge feature set and fatal data
m = pd.merge(feat, data, left_on=['STARTDATE','ADMIN1','COUNTRY'], right_on=['STARTDATE','ADMIN1','COUNTRY'], how='outer')

# COMMAND ----------

# check nans
print(m.isnull().sum())
m[pd.isnull(m).any(axis=1)]

# COMMAND ----------

# # these admin 1s are missing in the fata data because they had no events - fill with 0
# m[m.FATALSUM.isnull()]['ADMIN1'].unique()
m = m.fillna({'FATALSUM': 0})

# COMMAND ----------

# # these are erroneous - drop from data
# m[m.COUNTRY.isnull()]['ADMIN1'].unique()
m = m.dropna()

# COMMAND ----------

# select only needed columns
data = m.loc[:, ['STARTDATE', 'COUNTRY', 'ADMIN1', 'FATALSUM']].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Get escalation data

# COMMAND ----------

data

# COMMAND ----------

escalation_data = get_escalation_binary(data, 'FATALSUM', 'ADMIN1')
escalation_data[escalation_data['binary_escalation']== 1]

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. save it to pyspark 

# COMMAND ----------

# convert to spark dataframe
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
escalation_data = spark.createDataFrame(escalation_data)

DATABASE_NAME = 'news_media'
DATA_TABLE = 'horn_africa_acled_fatal_1w_10a10p_bin_gld'

# save in delta lake
escalation_data.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, DATA_TABLE))
