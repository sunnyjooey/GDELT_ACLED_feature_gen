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

# function to check if all continuous dates are in the data
# works for both weekly and daily intervals of dates

def check_all_dates(data, date_col, min_date, max_date):
    # unique dates in data
    dates_in_data = [row[date_col] for row in data.select(date_col).distinct().collect()]
    num_dates = len(dates_in_data)

    # weeks or days
    try: 
        diff = abs(((max_date - min_date) / 7).days - num_dates)
    except:
        max_date = dt.datetime.strptime(max_date, '%Y-%m-%d').date()
        min_date = dt.datetime.strptime(min_date, '%Y-%m-%d').date()
        diff = abs(((max_date - min_date) / 7).days - num_dates)
    if diff > 2:
        check_dates = [min_date + dt.timedelta(days=x) for x in range(num_dates)]
    else:
        check_dates = [min_date + dt.timedelta(weeks=x) for x in range(num_dates)]

    # dates that should be in data but isn't
    not_in_data = [str(x) for x in check_dates if x not in dates_in_data]
    # incongruencies in dates like None or wrong start/end date
    not_in_check = [str(x) for x in dates_in_data if x not in check_dates]
    if len(not_in_data) > 0:
        print('WARNING: the following dates should be but are not in the data:', not_in_data)
    if len(not_in_check) > 0:
        print('Warning: Check the following None or start/end date values:', not_in_check)
    

# COMMAND ----------

# cycle through all tables in the database
# check min and max dates
# make sure all continuous dates are in the data

for TAB in tables:
    table_name = TAB.name
    df = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{table_name};")

    # raw events data
    if 'events_brz' in table_name:
        df = df.withColumn('DATEADDED', F.to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
        df = df.withColumn('DATEADDED', F.to_date('DATEADDED'))
        min_date, max_date = df.select(F.min("DATEADDED"), F.max("DATEADDED")).first()
        print("TABLE", table_name, "goes from", min_date, "to", max_date)
        check_all_dates(df, 'DATEADDED', min_date, max_date)
        print()

    # raw embedding data
    elif 'embed_brz' in table_name:
        df = df.withColumn('date', F.regexp_replace('date', 'T', ' '))
        df = df.withColumn('date', F.regexp_replace('date', 'Z', ''))
        df = df.withColumn('date', F.to_timestamp('date', format='yyyy-MM-dd HH:mm:ss'))
        df = df.withColumn('date', F.to_date('date'))
        min_date, max_date = df.select(F.min("date"), F.max("date")).first()
        print("TABLE", table_name, "goes from", min_date, "to", max_date)
        check_all_dates(df, 'date', min_date, max_date)
        print()
    
    # no need to check error table
    elif ('error' in table_name) or ('cameo1_titlefill_sumfat_1w_popdense_conftrend_mod' == table_name):
        pass

    else:
        # check depending on date column
        cols = df.columns
        if 'DATEADDED' in cols:
            min_date, max_date = df.select(F.min("DATEADDED"), F.max("DATEADDED")).first()
            print("TABLE", table_name, "goes from", min_date, "to", max_date)
            check_all_dates(df, 'DATEADDED', min_date, max_date)
            print()
        elif ('STARTDATE' in cols) and ('ENDDATE' in cols):
            min_date, max_date = df.select(F.min("STARTDATE"), F.max("STARTDATE")).first()
            print("TABLE", table_name, "STARTDATE goes from", min_date, "to", max_date)
            check_all_dates(df, 'STARTDATE', min_date, max_date)
            min_date, max_date = df.select(F.min("ENDDATE"), F.max("ENDDATE")).first()
            print("TABLE", table_name, "ENDDATE goes from", min_date, "to", max_date)
            check_all_dates(df, 'ENDDATE', min_date, max_date)
            print()
        elif 'STARTDATE' in cols:
            min_date, max_date = df.select(F.min("STARTDATE"), F.max("STARTDATE")).first()
            print("TABLE", table_name, "STARTDATE goes from", min_date, "to", max_date)
            check_all_dates(df, 'STARTDATE', min_date, max_date)
            print()
        else:
            print("TABLE", table_name, "has no known data column.")
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
dt.date(2019, 12, 30).weekday()

# COMMAND ----------


