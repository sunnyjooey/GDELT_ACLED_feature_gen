from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

# import ACLED data function
def get_all_acled():
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
    return df_all

def get_one_co_data(df, cnty_code, admin_col):
    # sudan country code - filter first before converting to pandas
    df = df.filter(df.CountryFK==cnty_code)
    df = df.toPandas()
    # convert admin to category - make sure admins are not left out in groupby
    df[admin_col] = df[admin_col].astype('category')
    # create year-month column
    df['TimeFK_Event_Date'] = df['TimeFK_Event_Date'].apply(lambda x: dt.datetime.strptime(str(x),'%Y%m%d'))    
    return df