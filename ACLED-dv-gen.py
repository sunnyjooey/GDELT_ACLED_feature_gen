# Databricks notebook source
# MAGIC %md
# MAGIC ### Dependent variable construction 
# MAGIC
# MAGIC ### things to think about:

# COMMAND ----------

import pandas as pd
import numpy as np
import datetime as dt
from  itertools import product

# COMMAND ----------

DATABASE_NAME = 'news_media'
DATA_TABLE = 'horn_africa_acled_outcome_fatal_escbin_1w_1010_slv'

# COMMAND ----------

MIN = dt.datetime(2019, 12, 23, 0, 0, 0)
MAX = dt.datetime(2023, 4, 17, 0, 0, 0) 

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

    # filter dates after dec 23, 2019
    df = df[df['TimeFK_Event_Date'] >= MIN]

    return df

# COMMAND ----------

cnty_codes = [214, 227, 108, 104, 97, 224, 235, 175] #get all horn of africa countries, Sudan:214, South Sudan:227, Ethiopia:108, Eritrea:104, Djibouti:97,Somalia:224,Uganda:235,Kenya:175
df = get_data(df_all,'ACLED_Admin1',cnty_codes) 

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

# filter fatal data to match feature set's dates
data = data.loc[data['STARTDATE'] <= MAX]
# change to string to make merge easier
data['STARTDATE'] = data['STARTDATE'].astype(str)  

# COMMAND ----------

# get all admin1 to make sure none are missing
adm1 = spark.sql('SELECT * FROM news_media.horn_africa_gdelt_gsgembed_2w_a1_100_slv')
adm1 = adm1.select('ADMIN1').distinct().rdd.map(lambda r: r[0]).collect()

# Arta from DJ is no longer valid
print([x for x in adm1 if x not in list(data['ADMIN1'].unique())])
# 'Bahr el Ghazal' and 'Equatoria' are no longer valid
print([x for x in list(data['ADMIN1'].unique()) if x not in adm1])

# COMMAND ----------

# merge into data
d = pd.DataFrame(list(product(*[data.STARTDATE.unique(), adm1])), columns=['STARTDATE','ADMIN1'])
data = pd.merge(d, data, how='left')

# COMMAND ----------

# check nans
print(data.isnull().sum())
data[pd.isnull(data).any(axis=1)]

# COMMAND ----------

# fill in 
data.loc[data['ADMIN1']=='Rwampara', 'COUNTRY'] = 'UG' 
# fill in missing with 0 (for Rwampapra, UG)
data = data.fillna({'FATALSUM': 0})
# # these are erroneous - drop from data
data = data.dropna(subset=['COUNTRY'])

# COMMAND ----------

print(data.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Get escalation data

# COMMAND ----------

escalation_data = get_escalation_binary(data, 'FATALSUM', 'ADMIN1')
escalation_data = escalation_data.dropna() # this drops first time interval

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. save it to pyspark 

# COMMAND ----------

# convert to spark dataframe
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
escalation_data = spark.createDataFrame(escalation_data)

# COMMAND ----------

# save in delta lake
escalation_data.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, DATA_TABLE))
