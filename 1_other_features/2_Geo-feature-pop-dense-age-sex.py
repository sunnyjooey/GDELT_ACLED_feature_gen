# Databricks notebook source
!pip install fiona
!pip install rasterio
!pip install rasterstats
!pip install geopandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import re
import pandas as pd

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')
from db_table import DATABASE_NAME, GEO_POP_DENSE_AGESEX_TABLE

sys.path.append('./geo_util')
from g_util import test_compatibility, geo_plot, merge_geo

# COMMAND ----------

# file paths
shapefile_path = '/dbfs/FileStore/geospatial/shapefiles/'
tiff_path = '/dbfs/FileStore/geospatial/geotiff/age_sex_structure/'

# mapping
co_dict = {
    'djibouti':'DJ', 
    'eritrea':'ER', 
    'ethiopia':'ET', 
    'kenya':'KE', 
    'somalia':'SO',
    'south sudan':'OD', 
    'sudan':'SU', 
    'uganda':'UG'
}

# Get paths for the shape files and tiff files
countries = [key.replace(' ', '') for key in co_dict.keys()]
shp_files = os.listdir(shapefile_path)
shapefile_lst = [s for s in shp_files if any(cnty in s for cnty in countries)]
shapefile_lst = [os.path.join(shapefile_path, dr) for dr in shapefile_lst]                     
tiff_lst = next(os.walk(tiff_path), (None, None, []))[2]
tiff_lst = [os.path.join(tiff_path, file) for file in tiff_lst]

# Get male and female separately
tiff_lst_female = [f for f in tiff_lst if '_f_' in f]
tiff_lst_male = [f for f in tiff_lst if '_m_' in f]
tiff_lst = list(zip(tiff_lst_female, tiff_lst_male))

# COMMAND ----------

final_df = pd.DataFrame()
pattern = r"(?<=/)(global_\w_\d+_2020_1km)(?=\.tif)"

for tiff in tiff_lst:
    # combine the data
    df = pd.DataFrame()

    for shape in shapefile_lst:
        tif_f, tif_m = tiff
        # set up var name
        col_name1 = re.search(pattern, tif_f).group(0)
        col_name2 = re.search(pattern, tif_m).group(0)
        # merge df
        df1, df2 = merge_geo(shape, tif_f), merge_geo(shape, tif_m)
        df1.rename(columns={'mean': col_name1}, inplace=True)
        df1.loc[:, col_name2] = df2.loc[:, 'mean']
        # generate final df with target columns
        if 'ADM2_EN' not in df1.columns:
            df1.loc[:, 'ADM2_EN'] = None
        df_stats = df1.loc[:,['ADM0_EN', 'ADM1_EN', 'ADM2_EN', col_name1, col_name2]]
        df_stats.rename(columns={'ADM0_EN': 'country'}, inplace=True)
        # Stack different counrties into one table
        df = pd.concat([df, df_stats], axis=0)
    
    if len(final_df) == 0:
        final_df = df.copy()
    else:
        final_df = pd.concat([final_df, df.iloc[:,3:]], axis=1)


# COMMAND ----------

# Minor cleaning
final_df.loc[(final_df.loc[:, 'country'] == 'Sudan (the)'),'country'] = 'Sudan'

# Calculate aggregated stats for ages: 0-14, 15-64, 65+
agg_dict = {'female_0_14_2020': [], 'male_0_14_2020': [], 'female_15_64_2020': [], 'male_15_64_2020': [], 'female_65_plus_2020': [], 'male_65_plus_2020': []}

for tiff in tiff_lst:
    f, m = tiff
    var_f = re.search(pattern, f).group(0)
    var_m = re.search(pattern, m).group(0)
    for gender in [var_f, var_m]:
        if int(re.search(r"_(\d+)_2020", gender).group(1)) >= 0 and int(re.search(r"_(\d+)_2020", gender).group(1)) < 15:
            if gender.startswith('global_f'):
                agg_dict['female_0_14_2020'].append(gender)
            else:
                agg_dict['male_0_14_2020'].append(gender)
        
        elif int(re.search(r"_(\d+)_2020", gender).group(1)) < 65:
            if gender.startswith('global_f'):
                agg_dict['female_15_64_2020'].append(gender)
            else:
                agg_dict['male_15_64_2020'].append(gender)
        
        else:
            if gender.startswith('global_f'):
                agg_dict['female_65_plus_2020'].append(gender)
            else:
                agg_dict['male_65_plus_2020'].append(gender)

# Generate aggregate columns
for agg_cate, sub_cates in agg_dict.items():
    final_df.loc[:, agg_cate] = 0
    for sub_cate in sub_cates:
        final_df.loc[:, agg_cate] += final_df.loc[:, sub_cate]

# COMMAND ----------

adm_dict = {
    'Aj Jazirah':'Al Jazirah',
    'Abyei PCA':'Abyei',
    'Djiboutii':'Djibouti',
    'Benishangul Gumz':'Benshangul/Gumuz',
    "Murang'a":'Muranga',
    'Tadjoura':'Tadjourah',
    'South West Ethiopia':'South West',
    'Elgeyo-Marakwet':'Elgeyo Marakwet'
}

# modify var names
final_df['ADMIN1'] = final_df.apply(lambda row: row['ADM1_EN'] if row['ADM2_EN']==None else row['ADM2_EN'], axis=1)
final_df['ADMIN1'] = final_df.apply(lambda row: adm_dict[row['ADMIN1']] if row['ADMIN1'] in adm_dict else row['ADMIN1'], axis=1)
final_df.rename(columns={'country': 'COUNTRY'}, inplace=True)
final_df['COUNTRY'] = final_df.apply(lambda row: co_dict[row['COUNTRY'].lower()], axis=1)
final_df.reset_index(drop=True, inplace=True)
final_df.drop(['ADM1_EN','ADM2_EN'], axis=1, inplace=True)

# reorder cols
first_cols = ['COUNTRY', 'ADMIN1']
later_cols = [col for col in final_df.columns if col not in first_cols]
df = final_df.loc[:, first_cols+later_cols]

# COMMAND ----------

# convert
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df = spark.createDataFrame(df)

# COMMAND ----------

# save in delta lake
# this will write if the table does not exist, but throw an error if it does exist
df.write.mode('errorifexists').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, GEO_POP_DENSE_AGESEX_TABLE))
