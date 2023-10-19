# Databricks notebook source
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

from pyspark.sql.functions import regexp_replace, to_timestamp, to_date

# COMMAND ----------

events = spark.sql(f"SELECT * FROM news_media.horn_africa_gdelt_events_brz")
events = events.withColumn('DATEADDED', to_timestamp('DATEADDED', format='yyyyMMddHHmmss'))
events = events.withColumn('DATEADDED', to_date('DATEADDED'))
events = events.filter(events.IsRootEvent=='1')

# COMMAND ----------

embed = spark.sql(f"SELECT * FROM news_media.horn_africa_gdelt_gsgembed_brz")
embed = embed.withColumn('date', regexp_replace('date', 'T|Z', ''))
embed = embed.withColumn('date', F.to_timestamp('date', format='yyyy-MM-ddHH:mm:ss'))
embed = embed.withColumn('date', F.to_date('date'))

# COMMAND ----------

start_date = '2020-04-01'
end_date = '2020-04-05'

# COMMAND ----------

s = dt.datetime.strptime(start_date, '%Y-%m-%d')
e = dt.datetime.strptime(end_date, '%Y-%m-%d')

evnt = events.filter((events.DATEADDED >= s) & (events.DATEADDED <= e))

# COMMAND ----------

CO = 'SU'

shapefiles = {
    'DJ': '/dbfs/FileStore/df/shapefiles/djibouti_adm1/dji_admbnda_gadm_adm1_2022.shp',
    'ER': '/dbfs/FileStore/df/shapefiles/eritrea_adm1/eri_admbnda_adm1_gov_20200427.shp',
    'ET': '/dbfs/FileStore/df/shapefiles/ethiopia_adm1/eth_admbnda_adm1_csa_bofedb_2021.shp',
    'KE': '/dbfs/FileStore/df/shapefiles/kenya_adm1/ken_admbnda_adm1_iebc_20191031.shp',
    'SO': '/dbfs/FileStore/df/shapefiles/somalia_adm1/som_admbnda_adm1_ocha_20230308.shp',
    'OD': '/dbfs/FileStore/df/shapefiles/southsudan_adm1/ssd_admbnda_adm1_imwg_nbs_20221219.shp',
    'SU': '/dbfs/FileStore/df/shapefiles/sudan_adm1/sdn_admbnda_adm1_cbs_nic_ssa_20200831.shp',
    'UG': '/dbfs/FileStore/df/shapefiles/uganda_adm1/uga_admbnda_adm2_ubos_20200824.shp'
}

gdf = gpd.read_file(shapefiles[CO])

if CO == 'DJ':
    gdf.loc[gdf['ADM1_EN']=='Djiboutii', 'ADM1_EN'] = 'Djibouti'
    gdf.loc[gdf['ADM1_EN']=='Tadjoura', 'ADM1_EN'] = 'Tadjourah'
elif CO == 'ET':
    gdf.loc[gdf['ADM1_EN']=='Benishangul Gumz', 'ADM1_EN'] = 'Benshangul/Gumuz'
    gdf.loc[gdf['ADM1_EN']=='South West Ethiopia', 'ADM1_EN'] = 'South West'
elif CO == 'KE':
    gdf.loc[gdf['ADM1_EN']=="Murang'a", 'ADM1_EN'] = 'Muranga'
    gdf.loc[gdf['ADM1_EN']=='Elgeyo-Marakwet', 'ADM1_EN'] = 'Elgeyo Marakwet'
elif CO == 'SU':
    gdf.loc[gdf['ADM1_EN']=='Abyei PCA', 'ADM1_EN'] = 'Abyei'
    gdf.loc[gdf['ADM1_EN']=='Aj Jazirah', 'ADM1_EN'] = 'Al Jazirah'    

# COMMAND ----------

evnt = evnt.filter((evnt.ActionGeo_CountryCode==CO) |(evnt.Actor1Geo_CountryCode==CO) | (evnt.Actor2Geo_CountryCode==CO))

# COMMAND ----------


if CO == 'ER':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ER06'), 'ADM1_EN'] = 'Semienawi Keih Bahri'
elif CO == 'ET':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET53'), 'ADM1_EN'] = 'Tigray'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET49'), 'ADM1_EN'] = 'Gambela'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET47'), 'ADM1_EN'] = 'Benshangul/Gumuz'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='ET52'), 'ADM1_EN'] = 'Somali'
elif CO == 'KE':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='KE09'), 'ADM1_EN'] = 'Busia'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='KE07'), 'ADM1_EN'] = 'Migori'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='KE08'), 'ADM1_EN'] = 'Kajiado'

    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mtongwe, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mwache, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kilimani, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='English Point, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Vanga, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Diani Beach, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kombeni, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Shanzu, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Vuka, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kisite, Coast, Kenya'), 'ADM1_EN'] = 'Kwale' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Port Reitz, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Samaki, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Ungama Bay, Coast, Kenya'), 'ADM1_EN'] = 'Tana River'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kikambala, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Ishakani, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kiunga, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Shimanzi, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kinondo, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Tudor Creek, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Ras Kitau, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Englishpoint, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Portreitz, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Diani Beach Hotel, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kiangwe, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mkokoni, Coast, Kenya'), 'ADM1_EN'] = 'Lamu' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mandabay, Coast, Kenya'), 'ADM1_EN'] = 'Lamu' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Gazi Bay, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Shelly Beach, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Nyali Beach, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mvindeni, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Chale Island, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mtwapa Creek, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Jumba La Mtwana, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kizingoni, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Little Head, Coast, Kenya'), 'ADM1_EN'] = 'Lamu' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kilindini Harbour, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa' 
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Kilifi Creek, Coast, Kenya'), 'ADM1_EN'] = 'Kilifi'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Leisure Lodge, Coast, Kenya'), 'ADM1_EN'] = 'Kwale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mtongwe Ferry, Coast, Kenya'), 'ADM1_EN'] = 'Mombasa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Mkanda Channel, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['GEO_NAME']=='Tenewi, Coast, Kenya'), 'ADM1_EN'] = 'Lamu'
elif CO == 'SO':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO19'), 'ADM1_EN'] = 'Togdheer'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO03'), 'ADM1_EN'] = 'Bari'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SO09'), 'ADM1_EN'] = 'Lower Juba'
elif CO == 'OD':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='OD07'), 'ADM1_EN'] = 'Upper Nile'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='OD09'), 'ADM1_EN'] = 'Western Bahr el Ghazal'
elif CO == 'SU':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SU36'), 'ADM1_EN'] = 'Red Sea'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SU49'), 'ADM1_EN'] = 'South Darfur'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM1_EN'])) & (adm_gdf['ADMIN1']=='SU47'), 'ADM1_EN'] = 'West Darfur'
elif CO == 'UG':
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UG34'), 'ADM2_EN'] = 'Kabale'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UGD6'), 'ADM2_EN'] = 'Manafwa'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UG43'), 'ADM2_EN'] = 'Kisoro'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UGC4'), 'ADM2_EN'] = 'Bukwo'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UG59'), 'ADM2_EN'] = 'Ntungamo'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UG28'), 'ADM2_EN'] = 'Bundibugyo'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UGC9'), 'ADM2_EN'] = 'Isingiro'
    adm_gdf.loc[(pd.isna(adm_gdf['ADM2_EN'])) & (adm_gdf['ADMIN1']=='UG61'), 'ADM2_EN'] = 'Rakai'

# designate the rest to entire country
adm_gdf['ADM1_EN'].fillna(CO, inplace=True)

# COMMAND ----------

display(evnt)

# COMMAND ----------




before = s + dt.timedelta(-3)
after = e + dt.timedelta(2)

emb = embed.filter((embed.date >= before) & (embed.date <= after))

    events = events.dropDuplicates(['SOURCEURL'])
    events = events.toPandas()
