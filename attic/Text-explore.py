# Databricks notebook source
# MAGIC %md
# MAGIC ### ACLED notes explore

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

# MAGIC %md
# MAGIC ### GDELT title explore

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, to_date, regexp_replace
import string

events = spark.sql(f"SELECT * FROM news_media.horn_africa_gdelt_events_a1_slv")

embed = spark.sql(f"SELECT * FROM news_media.horn_africa_gdelt_gsgembed_brz")
embed = embed.withColumn('date', regexp_replace('date', 'T|Z', ''))
embed = embed.withColumn('date', to_timestamp('date', format='yyyy-MM-ddHH:mm:ss'))
embed = embed.withColumn('date', to_date('date'))

# COMMAND ----------


def get_gdelt(events, embed, CO, admin1, start_date, end_date):
    s = dt.datetime.strptime(start_date, '%Y-%m-%d')
    e = dt.datetime.strptime(end_date, '%Y-%m-%d')
    evt = events.filter((events.DATEADDED >= s) & (events.DATEADDED <= e))
    evt = evt.filter(((evt.COUNTRY == CO) & (evt.ADMIN1 == CO)) | (evt.ADMIN1 == admin1))

    before = s + dt.timedelta(-3)
    after = e + dt.timedelta(2)
    emb = embed.filter((embed.date >= before) & (embed.date <= after)).select('url', 'title', 'lang')
    mrg = evt.join(emb, evt.SOURCEURL==emb.url, 'left').select('DATEADDED', 'ADMIN1', 'COUNTRY', 'SOURCEURL', 'title', 'lang')

    coi = string.ascii_lowercase.index(CO[0].lower())
    adi = string.ascii_lowercase.index(admin1[0].lower())
    if coi > adi:
        asc_lst = [1, 1]
    else:
        asc_lst = [1, 0]

    mrg = mrg.dropDuplicates(['SOURCEURL']).orderBy('DATEADDED', 'ADMIN1', ascending=asc_lst)
    return mrg

# COMMAND ----------

sub = get_gdelt(events, embed, 'ET', 'Tigray', '2020-04-01', '2020-04-06')
display(sub)

# COMMAND ----------


