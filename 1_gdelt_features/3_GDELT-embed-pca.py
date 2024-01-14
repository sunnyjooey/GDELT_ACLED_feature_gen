# Databricks notebook source
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# COMMAND ----------

DATABASE_NAME = 'news_media' 
INPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_slv'

# COMMAND ----------

data = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{INPUT_TABLE_NAME}")
df = data.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA. Full dataset, LARGE N of components (500)

# COMMAND ----------

#dropping non-numerical features
d = df.drop(['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'], axis=1) 

# setting N of components to 500
pca = PCA(n_components=500)

# performing PCA
pca_result_500component = pca.fit_transform(d)

# bringing back non-numericals
non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]
pca_result_500component = pd.DataFrame(pca_result_500component)
pca_result_500component = pd.concat([non_numericals, pca_result_500component], axis=1)

#saving dataset
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_pca_500_slv'
pca_result_500component = spark.createDataFrame(pca_result_500component)
pca_result_500component.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME)) 

#pca_result_500component.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA. Full dataset, SMALL N of components (200)

# COMMAND ----------

#dropping non-numerical features
d = df.drop(['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'], axis=1) 

# setting N of components to 200
pca = PCA(n_components=200)

# performing PCA
pca_result_200component = pca.fit_transform(d)

# bringing back non-numericals
non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]
pca_result_200component = pd.DataFrame(pca_result_200component)
pca_result_200component = pd.concat([non_numericals, pca_result_200component], axis=1)

#saving dataset
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_pca_200_slv'
pca_result_200component = spark.createDataFrame(pca_result_200component)
pca_result_200component.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME)) 

#pca_result_200component.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA: preformed on different 't' separately, with N=120 components per type of 't'

# COMMAND ----------

t1s = df.iloc[:, 4:516]
t2s = df.iloc[:, 516:1028]
t3s = df.iloc[:, 1028:1540]
t4s = df.iloc[:, 1540:]

# keeping non-numerical features separately
non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]

# setting N of components to 120
pca = PCA(n_components=120)

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
t1 = 't-1'
t2 = 't-2'
t3 = 't-3'
t4 = 't-4'

for i in range(120):
    t = str(i) + '_' + t1
    cols.append(t)
for i in range(120):
    t = str(i) + '_' + t2
    cols.append(t)
for i in range(120):
    t = str(i) + '_' + t3
    cols.append(t)
for i in range(120):
    t = str(i) + '_' + t4
    cols.append(t)

pca_per_t_120comp.columns = cols

# inspect the result
#pca_per_t_120comp.head(10)

#saving dataset
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_pca_120_per_t_slv'
pca_per_t_120comp = spark.createDataFrame(pca_per_t_120comp)
pca_per_t_120comp.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME)) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA: preformed on different 't' separately, with N=50 components per type of 't'

# COMMAND ----------

t1s = df.iloc[:, 4:516]
t2s = df.iloc[:, 516:1028]
t3s = df.iloc[:, 1028:1540]
t4s = df.iloc[:, 1540:]

# keeping non-numerical features separately
non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]

# setting N of components to 50
pca = PCA(n_components=50)

# performing PCA
pca_50_t1 = pca.fit_transform(t1s)
pca_50_t2 = pca.fit_transform(t2s)
pca_50_t3 = pca.fit_transform(t3s)
pca_50_t4 = pca.fit_transform(t4s)

# bringing back non-numericals and concatenating pca results per each 't-X'
non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]
pca_50_t1 = pd.DataFrame(pca_50_t1)
pca_50_t2 = pd.DataFrame(pca_50_t2)
pca_50_t3 = pd.DataFrame(pca_50_t3)
pca_50_t4 = pd.DataFrame(pca_50_t4)

pca_per_t_50comp = pd.concat([non_numericals, pca_50_t1, pca_50_t2, pca_50_t3, pca_50_t4], axis=1)

# taking care of the columns names
cols = ['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']
t1 = 't-1'
t2 = 't-2'
t3 = 't-3'
t4 = 't-4'

for i in range(50):
    t = str(i) + '_' + t1
    cols.append(t)
for i in range(50):
    t = str(i) + '_' + t2
    cols.append(t)
for i in range(50):
    t = str(i) + '_' + t3
    cols.append(t)
for i in range(50):
    t = str(i) + '_' + t4
    cols.append(t)

pca_per_t_50comp.columns = cols

# inspect the result
# pca_per_t_50comp.head(10)

#saving dataset
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_pca_50_per_t_slv'
pca_per_t_50comp = spark.createDataFrame(pca_per_t_50comp)
pca_per_t_50comp.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME)) 

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

#dropping non-numerical features
d = df.drop(['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'], axis=1) 

# setting N of components to 383
pca = PCA(n_components=383)

# performing PCA
pca_result_383component = pca.fit_transform(d)

# bringing back non-numericals
non_numericals = df[['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY']]
pca_result_383component = pd.DataFrame(pca_result_383component)
pca_result_383component = pd.concat([non_numericals, pca_result_383component], axis=1)

#saving dataset
OUTPUT_TABLE_NAME = 'horn_africa_gdelt_gsgembed_1w_a1_8020_lag4_pca_383_slv'
pca_result_383component = spark.createDataFrame(pca_result_383component)
pca_result_383component.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, OUTPUT_TABLE_NAME)) 

#pca_result_383component.head(3)

# COMMAND ----------

# # Best number of components? part2
# d = df.drop(['STARTDATE', 'ENDDATE', 'ADMIN1', 'COUNTRY'], axis=1) 
# pca = PCA(n_components='mle') # Minka’s MLE is used to guess the dimension, aka 'automatic'
# pca_mle = pca.fit_transform(d)
# pca_mle[0].size
