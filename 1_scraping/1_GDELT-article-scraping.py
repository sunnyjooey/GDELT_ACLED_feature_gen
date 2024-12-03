# Databricks notebook source
!pip install git+https://github.com/codelucas/newspaper.git

# COMMAND ----------

# events data 
events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EVENT_PROCESS_TABLE}")
events = events.toPandas()

# COMMAND ----------

import time
import pandas as pd
from newspaper import Article
from pyspark.sql.types import StructType, StructField, StringType, DateType

# COMMAND ----------

# import variables
import sys
sys.path.append('../util')

from db_table import START_DATE, END_DATE, DATABASE_NAME, GDELT_EVENT_PROCESS_TABLE

# COMMAND ----------

# events data 
events = spark.sql(f"SELECT * FROM {DATABASE_NAME}.{GDELT_EVENT_PROCESS_TABLE}")
events = events.dropDuplicates(['SOURCEURL'])

# COMMAND ----------

all_events = events.toPandas()
all_events.shape

# COMMAND ----------

# initialize
start_time = time.time()
save_every_i = 1000
articles = []
urls = []
dates = []

# start index
start = 149000

for i, row in all_events.iterrows():
    if i > start:
        url = row['SOURCEURL']
        date = row['DATEADDED']
        article = Article(url)

        try:
            article.download()
            article.parse()
            article_text = article.text
            articles.append(article_text)
            urls.append(url)
            dates.append(date)
        except:
            print(i, 'exception')

        # save at every 1000 articles and last bits
        if (i%save_every_i==0) or (i==len(all_events)-1):
            # create pyspark dataframe
            texts = pd.DataFrame(zip(dates, urls, articles), columns=['DATEADDED','SOURCEURL','TEXT'])
            spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
            schema = [StructField('DATEADDED', DateType(), True), StructField('SOURCEURL', StringType(), True), StructField('TEXT', StringType(), True)]
            spdf = spark.createDataFrame(texts, StructType(schema))    
            spdf.write.mode('append').format('delta').saveAsTable("{}.{}".format(DATABASE_NAME, 'gdelt_events_cameo1_scraped'))

            # Print info
            end_time = time.time()
            print(f'====> Finished processing {i} after:', round((end_time - start_time) / 60, 2), 'minutes.')
            print('Number of scraped articles: ', len(texts))
            print('')

            # reset
            start_time = time.time()
            articles = []
            urls = []
            dates = []


# COMMAND ----------

# MAGIC %md
# MAGIC ====> Finished processing 1000 after: 24.32 minutes.
# MAGIC Number of scraped articles:  645
# MAGIC
# MAGIC ====> Finished processing 2000 after: 22.26 minutes.
# MAGIC Number of scraped articles:  830
# MAGIC
# MAGIC ====> Finished processing 149000 after: 21.28 minutes.
# MAGIC Number of scraped articles:  588
# MAGIC
# MAGIC ====> Finished processing 178566 after: 12.19 minutes.
# MAGIC Number of scraped articles:  363

# COMMAND ----------


