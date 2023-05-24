# Databricks notebook source
DATABASE_NAME = 'news_media'
EVTSLV_TABLE_NAME = 'horn_africa_gdelt_events_su_slv'
EMB_TABLE_NAME = 'horn_africa_gdelt_gsgembed_brz'
CO = 'SU'

# COMMAND ----------

evtslv = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EVTSLV_TABLE_NAME}")
emb = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{EMB_TABLE_NAME}")

# COMMAND ----------

# merge with embeddings
all_df = evtslv.join(emb, evtslv.SOURCEURL==emb.url, how='left')

# COMMAND ----------

display(all_df.limit(10))

# COMMAND ----------

cols = ['SQLDATE', 'ADMIN1_NAME'] + list(np.arange(512).astype(str))
embed_multiples = all_df.select(*cols)

# COMMAND ----------

import datetime as dt
embed_multiples['SQLDATE'] = embed_multiples['SQLDATE'].apply(lambda x: dt.datetime.strptime(str(x),'%Y%m%d'))

# COMMAND ----------

embed_multiples = embed_multiples[embed_multiples['SQLDATE'] >= dt.datetime(2020,1,1)]

# COMMAND ----------

freq = '3D'
co_frac = embed_multiples.groupby([pd.Grouper(key='SQLDATE', freq=freq), 'ADM1_EN']).size() / embed_multiples.groupby([pd.Grouper(key='SQLDATE', freq=freq)]).size()
idx = pd.IndexSlice
co_fracs = pd.DataFrame(co_frac).loc[idx[:, CO], :]
co_fracs

# COMMAND ----------

# cast as category to have all admin 1s even if nan
embed_multiples['ADMIN1_NAME'] = embed_multiples['ADMIN1_NAME'].astype('category')

# COMMAND ----------

# average by date and admin 1
avg = embed_multiples.groupby([pd.Grouper(key='SQLDATE', freq=freq), 'ADMIN1_NAME']).mean()

# COMMAND ----------

# unique dates and admin 1s
dates = avg.index.get_level_values(0).unique()
admins = avg.index.get_level_values(1).unique()
admins = [a for a in admins if a != CO]

# COMMAND ----------

# cycle through and replace with weighted averages
for d in dates:
    for a in admins:
        co_per = co_fracs.loc[idx[d,CO], 0]
        adm_per = 1 - co_per
        if avg.loc[idx[(d, a)], :].isnull().values.any():
            # if NaN, repace with country's news
            avg.loc[idx[d, a], :] = avg.loc[idx[d, CO], :]
        else:
            # if not, take a weighted average
            avg.loc[idx[d, a], :] = np.average(avg.loc[idx[(d, [a, CO])], :], weights=[adm_per, co_per], axis=0)


# COMMAND ----------

# drop CO rows
avg = avg.loc[idx[:, admins], :]
avg 

# COMMAND ----------


