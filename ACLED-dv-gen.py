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
    data['abs_change'] = data.groupby(admin1_col)[fat_column].diff().abs()

    # Percentage increase
    smoothing_factor = 1e-10
    data['pct_increase'] = (data['abs_change'] / (data.groupby(admin1_col)[fat_column].shift() + smoothing_factor)) * 100

    # Create a new column for binary_escalation
    data['binary_escalation'] = None

    # Apply thresholds for binary_escalation
    data.loc[(data[fat_column] >= 100) & (data['pct_increase'] >= 10), 'binary_escalation'] = 1
    data.loc[(data[fat_column] < 100) & (data['abs_change'] > 10), 'binary_escalation'] = 0

    # Return the modified DataFrame
    return data


# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Process data and plot

# COMMAND ----------

data = process_data(df)

# COMMAND ----------

feat = spark.sql("SELECT * FROM news_media.horn_africa_acled_confhist_2w_gld")
feat = feat.toPandas()

# COMMAND ----------

feat[feat['COUNTRY']=='OD']['ADMIN1'].unique()

# COMMAND ----------

dat = data.loc[data['TimeFK_Event_Date'] <= feat['STARTDATE'].max()]
dat['TimeFK_Event_Date'] = dat['TimeFK_Event_Date'].astype(str)  # makes merges easier

# COMMAND ----------

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

dat['CountryFK'] = dat['CountryFK'].map(country_keys)

# COMMAND ----------

f = feat.iloc[:, :4]
f.head()

# COMMAND ----------

m = pd.merge(f, dat, left_on=['STARTDATE','ADMIN1','COUNTRY'], right_on=['TimeFK_Event_Date','ACLED_Admin1','CountryFK'], how='outer')

# COMMAND ----------

m[pd.isnull(m).any(axis=1)]

# COMMAND ----------

m.isnull().sum()

# COMMAND ----------

m[m.ACLED_Admin1.isnull()]['ADMIN1'].unique()

# COMMAND ----------

m[(m.ADMIN1.isnull()) & (m.CountryFK!='SU') & (m.ACLED_Admin1!='Rwampara')]

# COMMAND ----------

SHAPEFILE = '/dbfs/FileStore/df/shapefiles/southsudan_adm1/ssd_admbnda_adm1_imwg_nbs_20221219.shp'

# COMMAND ----------

!pip install pysal
!pip install descartes

# COMMAND ----------

import geopandas as gpd
gdf = gpd.read_file(SHAPEFILE)

# COMMAND ----------

gdf

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Get escalation data

# COMMAND ----------

data

# COMMAND ----------

escalation_data =get_escalation_binary(data, 'ACLED_Fatalities', 'ACLED_Admin1')

# COMMAND ----------

escalation_data[escalation_data['binary_escalation']== 1]

# COMMAND ----------

# Group the data by country and calculate the total fatalities over time
grouped_df = escalation_data.groupby(['CountryFK', 'TimeFK_Event_Date']).agg({'ACLED_Fatalities': 'sum'}).reset_index()

##start_date = pd.to_datetime('2020-01-30')
##end_date = pd.to_datetime('2021-09-30')

#grouped_df = grouped_df[
    #(grouped_df['TimeFK_Event_Date'].dt.year.between(start_date.year, end_date.year)) &
    #(grouped_df['TimeFK_Event_Date'].dt.month.between(start_date.month, end_date.month))]

# Map the country codes to country names
grouped_df['CountryFK'] = grouped_df['CountryFK'].map(country_codes)
plt.figure(figsize=(30, 10)) 

# Plotting the fatalities over time by country
for country_code in grouped_df['CountryFK'].unique():
    country_data = grouped_df[grouped_df['CountryFK'] == country_code]
    plt.plot(country_data['TimeFK_Event_Date'], country_data['ACLED_Fatalities'], label=country_code)

# Filter escalation data based on start and end dates


# Markers for escalation binary
escalation_df = escalation_data[escalation_data['binary_escalation'] == 1]
# Filter escalation data based on start and end dates
#escalation_df = escalation_df[
    #(escalation_df['TimeFK_Event_Date'].dt.year.between(start_date.year, end_date.year)) &
    #(escalation_df['TimeFK_Event_Date'].dt.month.between(start_date.month, end_date.month))] 

plt.plot(escalation_df['TimeFK_Event_Date'], escalation_df['ACLED_Fatalities'], marker='o', linestyle='', color='black', label='Escalation')

plt.xlabel('Time')
plt.ylabel('Fatalities')
plt.title('Fatalities Over Time by Country')
plt.legend()
plt.show()

# COMMAND ----------

country_codes = {
    214: 'Sudan',
    227: 'South Sudan',
    108: 'Ethiopia',
    104: 'Eritrea',
    97: 'Djibouti',
    224: 'Somalia',
    235: 'Uganda',
    175: 'Kenya'
}

country_key_code = {
    'SU': 'Sudan',
    'OD': 'South Sudan',
    'ET': 'Ethiopia',
    'ER': 'Eritrea',
    'DJ': 'Djibouti',
    'SO': 'Somalia',
    'UG': 'Uganda',
    'KE': 'Kenya'
}


# COMMAND ----------

#add in country foreign keys: Sudan SU, South Sudan OD,  Ethiopia ET, Eritrea ER, Dijibouti DJ, Somalia SO, Uganda UG, Kenya KE
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

escalation_data['Country_Key'] = escalation_data['CountryFK'].map(country_keys)

# COMMAND ----------

escalation_data

# COMMAND ----------

######## PLOT TO DOUBLE CHECK ############


# Group the data by country and calculate the total fatalities over time
grouped_df = escalation_data.groupby(['Country_Key', 'TimeFK_Event_Date']).agg({'ACLED_Fatalities': 'sum'}).reset_index()

# Define start and end dates for filtering
#start_date = pd.to_datetime('2020-01-30')
#end_date = pd.to_datetime('2021-09-30')

# Filter grouped_df based on start and end dates
#grouped_df = grouped_df[
   # (grouped_df['TimeFK_Event_Date'].dt.year.between(start_date.year, end_date.year)) &
   # (grouped_df['TimeFK_Event_Date'].dt.month.between(start_date.month, end_date.month))]

# Map the country codes to country names
grouped_df['Country_Key'] = grouped_df['Country_Key'].map(country_key_code)

# Create the plot
plt.figure(figsize=(30, 10))

# Plotting the fatalities over time by country
for country_code in grouped_df['Country_Key'].unique():
    country_data = grouped_df[grouped_df['Country_Key'] == country_code]
    plt.plot(country_data['TimeFK_Event_Date'], country_data['ACLED_Fatalities'], label=country_code)

# Filter escalation data based on start and end dates
escalation_df = escalation_data[escalation_data['binary_escalation'] == 1]
#escalation_df = escalation_df[
   # (escalation_df['TimeFK_Event_Date'].dt.year.between(start_date.year, end_date.year)) &
   # (escalation_df['TimeFK_Event_Date'].dt.month.between(start_date.month, end_date.month))]

# Plot escalation events
plt.plot(escalation_df['TimeFK_Event_Date'], escalation_df['ACLED_Fatalities'],
         marker='o', linestyle='', color='black', label='Escalation')

plt.xlabel('Time')
plt.ylabel('Fatalities')
plt.title('Fatalities Over Time by Country')
plt.legend()
plt.show()


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



# COMMAND ----------

DATABASE_NAME= 'news_media'
INPUT_TABLE_NAME='horn_africa_acled_fatal_1w_10a10p_bin_gld'

# COMMAND ----------

TEST = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}") 

# COMMAND ----------

display(TEST)

# COMMAND ----------

TEST = TEST.toPandas()
TEST

# COMMAND ----------

CHECK_TEST =TEST[TEST['binary_escalation'] == 1]
CHECK_TEST

# COMMAND ----------

# Group the data by country and calculate the total fatalities over time
grouped_df = TEST.groupby(['CountryFK', 'TimeFK_Event_Date']).agg({'ACLED_Fatalities': 'sum'}).reset_index()

##start_date = pd.to_datetime('2020-01-30')
##end_date = pd.to_datetime('2021-09-30')

#grouped_df = grouped_df[
    #(grouped_df['TimeFK_Event_Date'].dt.year.between(start_date.year, end_date.year)) &
    #(grouped_df['TimeFK_Event_Date'].dt.month.between(start_date.month, end_date.month))]

# Map the country codes to country names
grouped_df['CountryFK'] = grouped_df['CountryFK'].map(country_codes)
plt.figure(figsize=(30, 10)) 

# Plotting the fatalities over time by country
for country_code in grouped_df['CountryFK'].unique():
    country_data = grouped_df[grouped_df['CountryFK'] == country_code]
    plt.plot(country_data['TimeFK_Event_Date'], country_data['ACLED_Fatalities'], label=country_code)

# Filter escalation data based on start and end dates


# Markers for escalation binary
escalation_df = TEST[TEST['binary_escalation'] == 1]
# Filter escalation data based on start and end dates
#escalation_df = escalation_df[
    #(escalation_df['TimeFK_Event_Date'].dt.year.between(start_date.year, end_date.year)) &
    #(escalation_df['TimeFK_Event_Date'].dt.month.between(start_date.month, end_date.month))] 

plt.plot(escalation_df['TimeFK_Event_Date'], escalation_df['ACLED_Fatalities'], marker='o', linestyle='', color='black', label='Escalation')

plt.xlabel('Time')
plt.ylabel('Fatalities')
plt.title('Fatalities Over Time by Country')
plt.legend()
plt.show()
