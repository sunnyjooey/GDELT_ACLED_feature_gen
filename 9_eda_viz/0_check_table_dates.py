# Databricks notebook source
import datetime as dt
import pyspark.sql.functions as F

# import variables
import sys
sys.path.append('../util')
from db_table import DATABASE_NAME, GDELT_ERROR_TABLE, GDELT_EVENT_TABLE, GDELT_EMBED_TABLE

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check dates in tables

# COMMAND ----------

# get list of tables in database
tables = spark.catalog.listTables(DATABASE_NAME)

# COMMAND ----------

for TAB in tables:
    table_name = TAB.name
    df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{table_name};")

    # raw events data
    if 'events_brz' in table_name:
        df = df.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
        df = df.withColumn('DATEADDED', F.to_date('DATEADDED'))
        min_date, max_date = df.select(F.min("DATEADDED"), F.max("DATEADDED")).first()
        print("TABLE", table_name, "goes from", min_date, "to", max_date)
        print()

    # raw embedding data
    elif 'embed_brz' in table_name:
        df = df.withColumn('date', F.regexp_replace('date', 'T', ' '))
        df = df.withColumn('date', F.regexp_replace('date', 'Z', ''))
        df = df.withColumn('date', F.to_timestamp('date', format='yyyy-MM-dd HH:mm:ss'))
        df = df.withColumn('date', F.to_date('date'))
        min_date, max_date = df.select(F.min("date"), F.max("date")).first()
        print("TABLE", table_name, "goes from", min_date, "to", max_date)
        print()
    
    # no need to check error table
    elif 'error' in table_name:
        pass

    else:
        # check depending on date column
        cols = df.columns
        if 'DATEADDED' in cols:
            min_date, max_date = df.select(F.min("DATEADDED"), F.max("DATEADDED")).first()
            print("TABLE", table_name, "goes from", min_date, "to", max_date)
            print()
        elif ('STARTDATE' in cols) and ('ENDDATE' in cols):
            min_date, max_date = df.select(F.min("STARTDATE"), F.max("STARTDATE")).first()
            print("TABLE", table_name, "STARTDATE goes from", min_date, "to", max_date)
            min_date, max_date = df.select(F.min("ENDDATE"), F.max("ENDDATE")).first()
            print("TABLE", table_name, "ENDDATE goes from", min_date, "to", max_date)
            print()
        elif 'STARTDATE' in cols:
            min_date, max_date = df.select(F.min("STARTDATE"), F.max("STARTDATE")).first()
            print("TABLE", table_name, "STARTDATE goes from", min_date, "to", max_date)
            print()
        else:
            print("TABLE", table_name, "has no known data colum.")
            print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check number of rows in data

# COMMAND ----------

# dates to check in between
CHECK_START = '2023-03-01'
CHECK_END = '2023-07-03'

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Events data table

# COMMAND ----------

# read in and filter to date
df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EVENT_TABLE};")
df = df.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
df = df.withColumn('DATEADDED', F.to_date('DATEADDED'))
df_sub = df.filter((df['DATEADDED'] >= dt.datetime.strptime(CHECK_START, '%Y-%m-%d').date()) & (df['DATEADDED'] < dt.datetime.strptime(CHECK_END, '%Y-%m-%d').date()))

# groupby and count by date
df_sub = df_sub.groupBy('DATEADDED').count().orderBy('DATEADDED')

# COMMAND ----------

display(df_sub)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Embeddings data table

# COMMAND ----------

# read in and filter to date
df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EMBED_TABLE};")
df = df.withColumn('date', F.regexp_replace('date', 'T', ' '))
df = df.withColumn('date', F.regexp_replace('date', 'Z', ''))
df = df.withColumn('date', F.to_timestamp('date', format='yyyy-MM-dd HH:mm:ss'))
df = df.withColumn('date', F.to_date('date'))
df_sub = df.filter((df['date'] >= dt.datetime.strptime(CHECK_START, '%Y-%m-%d').date()) & (df['date'] < dt.datetime.strptime(CHECK_END, '%Y-%m-%d').date()))

# groupby and count by date
df_sub1 = df_sub.groupBy('date').count().orderBy('date')

# drop duplicates in url and do another groupby count
df_sub2 = df_sub.orderBy('date').coalesce(1).dropDuplicates(subset = ['url'])
df_sub2 = df_sub2.groupBy('date').count().orderBy('date')

# COMMAND ----------

display(df_sub1)

# COMMAND ----------

display(df_sub2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Errors table

# COMMAND ----------

df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_ERROR_TABLE};")
df = df.withColumn('date', F.to_timestamp('date', format='yyyyMMddHHmmss'))
df = df.withColumn('date', F.to_date('date'))
df_sub = df.filter((df['date'] >= dt.datetime.strptime(CHECK_START, '%Y-%m-%d').date()) & (df['date'] < dt.datetime.strptime(CHECK_END, '%Y-%m-%d').date()))
df_sub = df_sub.groupBy(['date','data']).count().orderBy('date')

# COMMAND ----------

display(df_sub)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check the weekday of a date
# MAGIC Note: `0` is Monday

# COMMAND ----------

import datetime as dt
dt.date(2023, 5, 15).weekday()

# COMMAND ----------


