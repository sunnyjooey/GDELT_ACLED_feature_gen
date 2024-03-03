# Databricks notebook source
# MAGIC %md
# MAGIC **What**: This notebook creates PCA data ready for modeling from embeddings data. It is most appropriate for data with many variables, so run after notebook `2_GDELT-embed-process-lag`. 
# MAGIC
# MAGIC **How**: Set the variables in util/db_table.py. 
# MAGIC
# MAGIC **Note**: Note that there is no date querying. For now, conduct PCA on the full data. For later, make functions to save PCA models and apply on incoming data. For this reason, table names (for outputs) are defined within the notebook instead of in `db_table`. The writes to the data lake will not append; it will write if the table does not exist, but throw an error if the table does exist.

# COMMAND ----------

# import libraries
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import DATABASE_NAME, GDELT_EMBED_PROCESS_LAG_TABLE

# COMMAND ----------

# import data
data = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EMBED_PROCESS_LAG_TABLE}")
df = data.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA. Full dataset

# COMMAND ----------

def full_data_pca(df, n_components, verbose=True):
    # define output table name
    OUTPUT_TABLE_NAME = GDELT_EMBED_PROCESS_LAG_TABLE.replace('_gld', f'_pca_{n_components}_gld').replace('_slv', f'_pca_{n_components}_gld')

    # dropping non-numerical features
    d = df.drop(['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'], axis=1) 

    # setting N of components 
    pca = PCA(n_components=n_components)

    # performing PCA
    pca_result = pca.fit_transform(d)

    # bringing back non-numericals
    non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]
    pca_result = pd.DataFrame(pca_result)
    pca_result = pd.concat([non_numericals, pca_result], axis=1)

    if verbose:
        print(OUTPUT_TABLE_NAME)
        display(pca_result.head(20))

    try:
        # saving dataset - if it doesn't exist
        pca_result = spark.createDataFrame(pca_result)
        pca_result.write.mode('errorifexists').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))
        print(f'Saved {DATABASE_NAME}.{OUTPUT_TABLE_NAME}!')
    except Exception as e:
        print(e)


# COMMAND ----------

### LARGE NUMBER OF COMPONENTS ###
full_data_pca(df, 500)

# COMMAND ----------

### SMALL NUMBER OF COMPONENTS ###
full_data_pca(df, 200)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA: preformed on different 't' separately

# COMMAND ----------

# Note: this function assumes 4 lags, further functionalization needed for other number of lags

def separate_t_pca(df, n_components, verbose=True):
    # define output table name
    OUTPUT_TABLE_NAME = GDELT_EMBED_PROCESS_LAG_TABLE.replace('_gld', f'_pca_{n_components}_per_t_gld').replace('_slv', f'_pca_{n_components}_per_t_gld')

    # split data by t
    t1s = df.loc[:, [x for x in df.columns if 't-1' in x]]
    t2s = df.loc[:, [x for x in df.columns if 't-2' in x]]
    t3s = df.loc[:, [x for x in df.columns if 't-3' in x]]
    t4s = df.loc[:, [x for x in df.columns if 't-4' in x]]

    # keeping non-numerical features separately
    non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]

    # setting N of components
    pca = PCA(n_components=n_components)

    # performing PCA
    pca_120_t1 = pca.fit_transform(t1s)
    pca_120_t2 = pca.fit_transform(t2s)
    pca_120_t3 = pca.fit_transform(t3s)
    pca_120_t4 = pca.fit_transform(t4s)

    # bringing back non-numericals and concatenating pca results per each 't-X'
    non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]
    pca_120_t1 = pd.DataFrame(pca_120_t1)
    pca_120_t2 = pd.DataFrame(pca_120_t2)
    pca_120_t3 = pd.DataFrame(pca_120_t3)
    pca_120_t4 = pd.DataFrame(pca_120_t4)

    pca_per_t_120comp = pd.concat([non_numericals, pca_120_t1, pca_120_t2, pca_120_t3, pca_120_t4], axis=1)

    # taking care of the columns names
    cols = ['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']
    cols.extend([f'{i}_t-1' for i in range(n_components)])
    cols.extend([f'{i}_t-2' for i in range(n_components)])
    cols.extend([f'{i}_t-3' for i in range(n_components)])
    cols.extend([f'{i}_t-4' for i in range(n_components)])
    pca_per_t_120comp.columns = cols

    if verbose:
        print(OUTPUT_TABLE_NAME)
        display(pca_per_t_120comp.head(20))

    try:
        # saving dataset - if it doesn't exist
        pca_per_t_120comp = spark.createDataFrame(pca_per_t_120comp)
        pca_per_t_120comp.write.mode('errorifexists').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))
        print(f'Saved {DATABASE_NAME}.{OUTPUT_TABLE_NAME}!')
    except Exception as e:
        print(e)

# COMMAND ----------

### LARGE NUMBER SEPARATE COMPONENTS ###
separate_t_pca(df, 120)

# COMMAND ----------

### SMALL NUMBER SEPARATE COMPONENTS ###
separate_t_pca(df, 50)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Best number of components?

# COMMAND ----------

d = df.drop(['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'], axis=1) 
pca = PCA(n_components=0.8, svd_solver='full') # variance greater than 80% is explained
pca_80_variance = pca.fit_transform(d)
# check number of components that explains 80% of variability
pca_80_variance[0].size

# COMMAND ----------

# PCA. Full dataset, N of components (383) - 80% of variance kept
n_components = 383

# output table name suggested: GDELT_EMBED_PROCESS_LAG_TABLE.replace('_gld', f'_pca_{n_components}_gld').replace('_slv', f'_pca_{n_components}_gld')
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_pca_383_slv'

# dropping non-numerical features
d = df.drop(['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'], axis=1) 

# setting N of components to 383
pca = PCA(n_components=n_components)

# performing PCA
pca_result_383component = pca.fit_transform(d)

# bringing back non-numericals
non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]
pca_result_383component = pd.DataFrame(pca_result_383component)
pca_result_383component = pd.concat([non_numericals, pca_result_383component], axis=1)

try:
    # saving dataset - if it doesn't exist
    pca_result_383component = spark.createDataFrame(pca_result_383component)
    pca_result_383component.write.mode('errorifexists').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME))
    print(f'Saved {DATABASE_NAME}.{OUTPUT_TABLE_NAME}!')
except Exception as e:
    print(e)
