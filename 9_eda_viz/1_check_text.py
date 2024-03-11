# Databricks notebook source
# read in modeling data
outcome = spark.sql('SELECT * FROM news_media.horn_africa_acled_outcome_fatal_escbin_1w_pct_v2_slv')
title_text = spark.sql('SELECT * FROM horn_africa_forecast.gdelt_events_cameo1_title_fill_1w_gld')

df = title_text.join(outcome, (outcome.STARTDATE==title_text.ENDDATE) & (outcome.ADMIN1==title_text.ADMIN1) & (outcome.COUNTRY==title_text.COUNTRY)).drop(title_text.STARTDATE).drop(title_text.ENDDATE).drop(title_text.ADMIN1).drop(title_text.COUNTRY)

# COMMAND ----------

import datetime as dt
from pyspark.sql import functions as F

# COMMAND ----------

# filter by outcome data, print text for checking
def filter_show(df, filter_dict):
    df_filter = df
    for key, val in filter_dict.items():
        df_filter = df_filter.filter(df[key] >= val)
    df_filter = df_filter.toPandas()
    for i, row in df_filter.iterrows():
        start_date = row['STARTDATE']
        adm1 = row['ADMIN1']
        co = row['COUNTRY']
        print(f"{start_date}, {adm1}, {co}; fatalities: {int(row['FATALSUM'])}, %increase: {round(row['pct_increase'], 1)}")
        print("The previous 1 week's titles")
        print(row['TITLE'])
        print('-------------------------------')

        # get previous time interval info
        prev_date = str((dt.datetime.strptime(start_date, '%Y-%m-%d') - dt.timedelta(weeks=1)).date())        
        df_sub = df.filter((df['STARTDATE']==prev_date) & (df['ADMIN1']==adm1) & (df['COUNTRY']==co))
        print(f"The week prior's fatalities: {int(df_sub.select(F.col('FATALSUM')).first().FATALSUM)}")
        print("The week prior's titles")
        print(df_sub.select(F.col('TITLE')).first().TITLE)
        print()
        print('+++++++++++++++++++++++++++++++')
        

# COMMAND ----------

filter_show(df, {'FATALSUM': 50, 'pct_increase': 10000})

# COMMAND ----------


