# Databricks notebook source
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

import datetime as dt
from pyspark.sql.types import DateType, StringType
from pyspark.sql.functions import col, udf, create_map, lit

co = {
    '214': 'Sudan',
    '227': 'S. Sudan',
    '108': 'Ethiopia',
    '104': 'Eritrea',
    '97': 'Djibouti',
    '224': 'Somalia',
    '235': 'Uganda',
    '175': 'Kenya'
}

def get_data(df, admin1, start_date, end_date):
    func =  udf (lambda x: dt.datetime.strptime(str(x), '%Y%m%d'), DateType())
    df = df.withColumn('TimeFK_Event_Date', func(col('TimeFK_Event_Date'))) 

    s = dt.datetime.strptime(start_date, '%Y-%m-%d')
    e = dt.datetime.strptime(end_date, '%Y-%m-%d')

    df = df.filter((df.ACLED_Admin1==admin1) & (df.TimeFK_Event_Date >= s) & (df.TimeFK_Event_Date <= e)).orderBy('TimeFK_Event_Date')
    df = df.select('TimeFK_Event_Date', 'CountryFK', 'ACLED_Admin1', 'ACLED_Event_Type','ACLED_Event_SubType','ACLED_Fatalities', 'ACLED_Notes')
    df = df.withColumn('CountryFK', df['CountryFK'].cast(StringType()))
    df = df.replace(to_replace=co, subset=['CountryFK'])

    return df

# COMMAND ----------

sub = get_data(df_all, 'Afar', '2021-07-10', '2021-07-17')
display(sub)

# COMMAND ----------


