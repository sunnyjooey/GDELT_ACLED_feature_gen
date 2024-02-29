# Databricks notebook source
import pyspark.sql.functions as F
from db_table import DATABASE_NAME

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
