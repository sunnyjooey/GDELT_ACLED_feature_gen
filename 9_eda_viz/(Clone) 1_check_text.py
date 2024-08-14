# Databricks notebook source
# import variables
import sys
sys.path.append('../util')
from db_table import START_DATE, END_DATE, DATABASE_NAME, GDELT_TITLE_FILL_TABLE, ACLED_OUTCOME_TABLE, COUNTRY_KEYS

# COMMAND ----------

outcome = spark.sql(f'SELECT * FROM {DATABASE_NAME}.{ACLED_OUTCOME_TABLE}')

# COMMAND ----------

o = outcome.filter(outcome['ADMIN1']=='Oromia')
o = o.toPandas()

# COMMAND ----------

display(o)

# COMMAND ----------

title_text = spark.sql('SELECT * FROM horn_africa_forecast.gdelt_events_cameo1_title_fill_1w_gld')
t = title_text.filter(title_text['ADMIN1']=='Oromia')

# COMMAND ----------

display(t)

# COMMAND ----------

# read in modeling data
outcome = spark.sql('SELECT * FROM news_media.horn_africa_acled_outcome_fatal_escbin_1w_pct_v2_slv')
title_text = spark.sql('SELECT * FROM horn_africa_forecast.gdelt_events_cameo1_title_fill_1w_gld')

# df = title_text.join(outcome, (outcome.STARTDATE==title_text.ENDDATE) & (outcome.ADMIN1==title_text.ADMIN1) & (outcome.COUNTRY==title_text.COUNTRY)).drop(title_text.STARTDATE).drop(title_text.ENDDATE).drop(title_text.ADMIN1).drop(title_text.COUNTRY)

# COMMAND ----------

import datetime as dt
from dateutil.relativedelta import relativedelta
from pyspark.sql import functions as F

# COMMAND ----------

# filter by outcome data, print text for checking
def filter_show(outcome, title_text, filter_dict):
    df_filter = outcome
    for key, val in filter_dict.items():
        df_filter = df_filter.filter(outcome[key] >= val)
    df_filter = df_filter.toPandas()
    for i, row in df_filter.iterrows():
        start_date = row['STARTDATE']
        adm1 = row['ADMIN1']
        co = row['COUNTRY']
        print(f"{start_date}, {adm1}, {co}; fatalities: {int(row['FATALSUM'])}, %increase: {round(row['pct_increase'], 1)}")
        print()
        prev_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date() + relativedelta(weeks=-1)
        prev_fatal = outcome.filter((outcome['STARTDATE']==prev_date) & (outcome['ADMIN1']==adm1)).select(F.col('FATALSUM')).first().FATALSUM
        prev_title = title_text.filter((title_text['STARTDATE']==prev_date) & (title_text['ADMIN1']==adm1))
        print(f'{prev_date} PREVIOUS 1 WEEK FATALITIES {prev_fatal} AND TITLES:')
        print(prev_title.select(F.col('TITLE')).first().TITLE)
        print()
        print()

        curr_title = title_text.filter((title_text['STARTDATE']==start_date) & (title_text['ADMIN1']==adm1))
        print(f"{start_date} CURRENT 1 WEEK FATALITIES {int(row['FATALSUM'])} AND TITLES:")
        print(curr_title.select(F.col('TITLE')).first().TITLE)
        print()
        print()

        next_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date() + relativedelta(weeks=1)
        next_fatal = outcome.filter((outcome['STARTDATE']==next_date) & (outcome['ADMIN1']==adm1)).select(F.col('FATALSUM')).first().FATALSUM
        next_title = title_text.filter((title_text['STARTDATE']==next_date) & (title_text['ADMIN1']==adm1))
        print(f'{next_date} NEXT 1 WEEK FATALITIES {next_fatal} AND TITLES:')
        print(next_title.select(F.col('TITLE')).first().TITLE)
        print()
        print('===============================')

        # next_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date() + relativedelta(weeks=1)
        # print(f"The following {next_date} 1 week's titles")
        # next_row = df.filter((df['STARTDATE']==next_date) & (df['ADMIN1']==adm1) & (df['COUNTRY']==co))
        # print(f"The week following's fatalities: {int(next_row.select(F.col('FATALSUM')).first().FATALSUM)}")
        # print("The week following's titles")
        # print(next_row.select(F.col('TITLE')).first().TITLE)
        # print()
        # print('===============================')

        # # get previous time interval info
        # prev_date = str((dt.datetime.strptime(start_date, '%Y-%m-%d') - dt.timedelta(weeks=1)).date())        
        # df_sub = df.filter((df['STARTDATE']==prev_date) & (df['ADMIN1']==adm1) & (df['COUNTRY']==co))
        # print(f"The week prior's {prev_date} fatalities: {int(df_sub.select(F.col('FATALSUM')).first().FATALSUM)}")
        # print("The week prior's titles")
        # print(df_sub.select(F.col('TITLE')).first().TITLE)
        # print()
        # print('+++++++++++++++++++++++++++++++')

        

# COMMAND ----------

filter_show(outcome, title_text, {'FATALSUM': 50, 'pct_increase': 10000})

# COMMAND ----------


