# Databricks notebook source
# MAGIC %md
# MAGIC **Note**: This notebook downloads global age-sex disaggregated tiff files from WorldPop. Set the `FILE_PATH` to a location in the DBFS before running the notebook. `2_Geo-feature-pop-dense-age-sex` notebok is dependent on this notebook.

# COMMAND ----------

!pip install lxml
!pip install cssselect

# COMMAND ----------

import requests
import lxml
from lxml import html
from lxml.cssselect import CSSSelector
from io import BytesIO
import os

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Downloading

# COMMAND ----------

URL = 'https://hub.worldpop.org/geodata/summary?id=24798'
FILE_PATH =  "dbfs:/FileStore/geospatial/geotiff/age_sex_structure/"

def req_data_links(source_web_url):
    """
    Fetch links for the datasets
    """

    # Scrape source webpage
    webpage = requests.get(URL)
    text = webpage.content
    # Parse the returned page
    parsed_text = html.fromstring(text)
    selector = CSSSelector('a.mt-3')
    elements = selector(parsed_text)

    return [element.get('href') for element in elements]


# COMMAND ----------

links = req_data_links(URL)
# create target directory to store files
if FILE_PATH not in dbutils.fs.ls("dbfs:/FileStore/geospatial/geotiff/"):
    dbutils.fs.mkdirs(FILE_PATH)
    print(f"Created directory: {FILE_PATH}")
else:
    print(f"Directory {FILE_PATH} already exists.")

# COMMAND ----------

for link in links:
    filename = link.split('/')[-1] # Extract filename from the link
    dbfs_path = FILE_PATH + filename

    if not os.path.exists(("/" + dbfs_path).replace(":", "")):
        try:
            # Download the file
            local_path = f"/tmp/{filename}"
        
            with requests.get(link, stream=True) as r:
                r.raise_for_status() # Raise an exception for HTTP errors

                # Write to local file system in chunks
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
                print(f'The {filename} has been downloaded')

                # Copy to DBFS
                dbutils.fs.cp(f"file://{local_path}", dbfs_path)
                print(f'It is saved to {dbfs_path}')
            
                # Remove the local file
                os.remove(local_path)

        except Exception as e:  
            print(f"Error processing link {link}. Error: {e}")
            continue

