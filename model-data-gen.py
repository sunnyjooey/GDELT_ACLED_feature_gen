# Databricks notebook source
out = spark.sql("SELECT * FROM news_media.horn_africa_acled_fatal_1w_10a10p_bin_gld")

# COMMAND ----------

conf = spark.sql("SELECT * FROM news_media.horn_africa_acled_confhist_2w_gld")

# COMMAND ----------

su = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_su_2w_slv")
od = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_od_2w_slv")
et = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_et_2w_slv")
er = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_er_2w_slv")
dj = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_dj_2w_slv")
so = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_so_2w_slv")
ug = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_ug_2w_slv")
ke = spark.sql("SELECT * FROM news_media.horn_africa_gdelt_gsgembed_ke_2w_slv")

# COMMAND ----------

display(out)

# COMMAND ----------

display(conf)

# COMMAND ----------

display(emb.head(600))

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame

emb = [su, od, et, er, dj, so, ug, ke]
emb = reduce(DataFrame.unionAll, emb)

# COMMAND ----------

from pyspark.sql.functions import date_format, col

emb = emb.withColumn("ENDDATE",  date_format(col("ENDDATE"), "yyyy-MM-dd"))

# COMMAND ----------

conf = conf.withColumn("STARTDATE",  date_format(col("STARTDATE"), "yyyy-MM-dd"))

# COMMAND ----------

conf.count()

# COMMAND ----------

emb.count()

# COMMAND ----------

display(emb)

# COMMAND ----------

m1 = conf.join(emb, conf.STARTDATE==emb.ENDDATE, how='left')

# COMMAND ----------

display(m1)

# COMMAND ----------

m1.count()

# COMMAND ----------


