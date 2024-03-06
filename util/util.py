import datetime as dt
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


def get_one_co_data(df, cnty_code, admin_col='ACLED_Admin1', time_col='TimeFK_Event_Date'):
    # sudan country code - filter first before converting to pandas
    df = df.filter(df.CountryFK==cnty_code)
    # drop columns difficult to convert
    df = df.drop(*['ACLED_Latitude', 'ACLED_Longitude'])
    # Convert to pandas dataframe
    df = df.toPandas()
    # convert admin to category - make sure admins are not left out in groupby
    df[admin_col] = df[admin_col].astype('category')
    # create year-month column
    df[time_col] = df[time_col].apply(lambda x: dt.datetime.strptime(str(x),'%Y%m%d'))    
    return df


def get_cnty_date_data(df, cnty_codes, start_date, end_date, admin_col='ACLED_Admin1', time_col='TimeFK_Event_Date'):
    # Filter by country
    df = df.filter(df.CountryFK.isin(cnty_codes))
    # drop columns difficult to convert
    df = df.drop(*['ACLED_Latitude', 'ACLED_Longitude'])
    # Convert to pandas dataframe
    df = df.toPandas()
    # Convert admin to category
    df[admin_col] = df[admin_col].astype('category')
    # Create year-month column
    df[time_col] = df[time_col].apply(lambda x: dt.datetime.strptime(str(x), '%Y%m%d'))
    # filter dates 
    df = df[(df[time_col] >= start_date) & (df[time_col] < end_date)]
    return df
