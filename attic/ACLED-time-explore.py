# Databricks notebook source
!pip install scikit-optimize
!pip install xgboost

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

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

def get_data(df, cnty_code, year_lst):
    # sudan country code
    df = df.filter(df.CountryFK==cnty_code) 
    df = df.toPandas()
    # create year-month column
    df['YearMonth'] = df['TimeFK_Event_Date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6])    
    # subset to particular years
    df_sub = df[df['ACLED_Year'].isin(year_lst)]
    print(f'The full dataset has {df.shape[0]} rows.')
    print(f'The year subset dataset has {df_sub.shape[0]} rows.')
    
    return df, df_sub

# COMMAND ----------

def make_lagged_features(data, num_lags, date_col, admin_col, event_type_col, value_col, agg_func):
    # each row is a time unit (month), columns are admin x event type
    piv = pd.pivot_table(data, index=date_col, columns=[admin_col, event_type_col], values=value_col, aggfunc=agg_func)
    piv.fillna(0, inplace=True)
    
    lagged_data = []
    t = np.arange(1, num_lags+1)[::-1]
    for i in range(num_lags):
        shifti = piv.shift(i*-1)
        shifti.columns = pd.MultiIndex.from_tuples([(col[0], f'{col[1]}_{value_col}_t-{t[i]}') for col in piv.columns])
        lagged_data.append(shifti)
    # concat column-wise    
    all_lagged_data = pd.concat(lagged_data, axis=1)
    # drop the bottom rows (most recent months), and reindex to 'outcome' month
    date_idx = all_lagged_data.index[num_lags: ]
    all_lagged_data = all_lagged_data[ :(-1 * num_lags)]
    all_lagged_data.index = date_idx
    # sort the columns
    all_lagged_data = all_lagged_data.reindex(sorted(all_lagged_data.columns), axis=1)
    
    # make long
    admin_levs = data[admin_col].unique()
#     # this restacking does not work correctly, good for creating bad data for model checking
#     stacked = pd.concat([all_lagged_data[admin] for admin in admin_levs], axis=0)
#     new_ind = pd.MultiIndex.from_product([all_lagged_data.index, admin_levs], names=['time','admin'])
#     stacked.index = new_ind
    stacked = pd.DataFrame()
    for admin in admin_levs:
        adm = all_lagged_data[admin]
        adm.index = pd.MultiIndex.from_product([adm.index, [admin]])
        stacked = pd.concat([stacked, adm], axis=0)
    return stacked

# COMMAND ----------

def get_months_since(col, m_since_num):
    boolean_array = np.asarray(col >= m_since_num)
    if boolean_array.sum() == 0:
        months_since = np.array([len(boolean_array)] * len(boolean_array))
    else:
        nz = np.nonzero(boolean_array)
        mat = np.arange(len(boolean_array)).reshape(-1, 1) - nz
        mat[mat < 1] = len(boolean_array)
        months_since = np.min(mat, axis=1)
    return months_since


def get_months_since_df(data, m_since_lst, date_col, admin_col, event_type_col, value_col, start_year, num_lags):
    fatal_piv = pd.pivot_table(data, index=date_col, columns=[admin_col, event_type_col], values=value_col, aggfunc='sum')
    fatal_piv.fillna(0, inplace=True)

    cols = pd.MultiIndex.from_product([data[admin_col].unique(), data[event_type_col].unique()])
    months_since_data = pd.DataFrame()
    for m_num in m_since_lst:
        for col in cols:
            colname = (col[0], col[1] + f'_since_{m_num}_death')  
            if col in fatal_piv.columns:
                months_since_data[colname] = get_months_since(fatal_piv[col], m_num)
            else:
                months_since_data[colname] = np.array([fatal_piv.shape[0]] * fatal_piv.shape[0])

    months_since_data.index = fatal_piv.index
    month = 1 + num_lags
    assert month < 13, "Pick a lag that is less than one year."        
    start_year_month = f'{start_year}-{month:02}'
    idx = list(months_since_data.index).index(start_year_month)
    months_since_data = months_since_data.iloc[idx: , : ]
    months_since_data.columns = pd.MultiIndex.from_tuples(months_since_data.columns)

    admin_levs = data[admin_col].unique()
    stacked_data = pd.DataFrame()
    for admin in admin_levs:
        adm = months_since_data[admin]
        adm.index = pd.MultiIndex.from_product([adm.index, [admin]])
        stacked_data = pd.concat([stacked_data, adm], axis=0)
        
    return stacked_data

# COMMAND ----------

def get_xy_df(data_lst, date_col, admin_col, outcome_col, agg_func):
    X = pd.concat(data_lst, axis=1)
    X.fillna(0, inplace=True) 
    y = df_sub.groupby([date_col, admin_col]).agg({outcome_col: agg_func})
    Xy = pd.concat([X, y], axis=1)
    Xy[outcome_col].fillna(0, inplace=True)
    Xy.dropna(how='any', inplace=True) 
    X = Xy.drop(outcome_col, axis=1)
    y = Xy[outcome_col]
    return X, y

def split_train_test(X, y, prop):
    time_units = X.index.unique(level=0)
    idx = round(len(time_units) * prop)
    train_times = time_units[:idx]
    test_times = [t for t in time_units if t not in train_times]
    
    X_train = X.loc[train_times, :]
    X_test = X.loc[test_times, :]
    y_train = y.loc[train_times]
    y_test = y.loc[test_times]
    return X_train, X_test, y_train, y_test
    

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV 
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Modeling function
params_xgb = {
        'booster':['gblinear'],
        'objective': ['reg:squarederror'],
        'lambda':(0.00001,10),
        'alpha':(0.00001,10),
        'feature_selector':['cyclic','shuffle']
}

params_rf = {
        'n_estimators': (5,500),
        'max_features': ['auto','sqrt'],
        'max_depth': (2,50),
        'min_samples_split': (2,10),
        'min_samples_leaf': (1,7),
        'bootstrap': ["True","False"]
}

xgb = BayesSearchCV(
        XGBRegressor(),
        params_xgb,
        n_iter=50,
        cv=5,
        random_state=42)

rf = BayesSearchCV(
        RandomForestRegressor(),
        params_rf,
        n_iter=50,
        cv=5,
        random_state=42)

def fit_model(X_train, y_train, search_cv):
    # model
    fitted_model = search_cv.fit(X_train, y_train)
    pred = fitted_model.predict(X_test)

    # results
    res = pd.DataFrame(fitted_model.cv_results_)
    return fitted_model, pred, res

# COMMAND ----------

all_res = pd.DataFrame()
# big loop
for start_year in [2015, 2016, 2017, 2018, 2019, 2020]:
    years_in = [str(x) for x in np.arange(start_year, 2023)]
    df, df_sub = get_data(df_all, 214, years_in)
    
    for t_min in [1, 2, 3, 4, 5]:
        counts = make_lagged_features(df_sub, t_min, 'YearMonth', 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_PK', 'count')
        fatal = make_lagged_features(df_sub, t_min, 'YearMonth', 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')
        
        for m_lst in [([1,5,50]), [1,5,10], [1,5,20], [1,5,10,20,50]]:
            mon_dat = get_months_since_df(df, m_lst, 'YearMonth', 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', start_year, t_min)
            X, y = get_xy_df([counts, fatal, mon_dat], 'YearMonth', 'ACLED_Admin1', 'ACLED_Fatalities', 'sum')
            X_train, X_test, y_train, y_test = split_train_test(X, y, .75)
            
            for search_cv in ['xgb', 'rf']:
                if search_cv == 'xgb':
                    model, predictions, results = fit_model(X_train, y_train, xgb)
                else:
                    model, predictions, results = fit_model(X_train, y_train, rf)
                results['start year'] = start_year
                results['lags'] = t_min
                results['months since'] = ('-').join([str(x) for x in m_lst])
                results['algos'] = search_cv
                best = results[results['rank_test_score']==1]
                all_res = pd.concat([all_res, best], axis=0)
                print(start_year, t_min, m_lst, search_cv)
                print("pred RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
                print("pred R2:", r2_score(y_test, predictions))
                print()
                

# COMMAND ----------

all_res

# COMMAND ----------

all_res.to_csv('/dbfs/FileStore/df/acled/sudan.csv', index=False)

# COMMAND ----------



# COMMAND ----------

df_sub.groupby(['ACLED_Interaction']).agg({'ACLED_Fatalities':[np.mean, np.sum]}).sort_values(('ACLED_Fatalities','sum'), ascending=False)

# COMMAND ----------

df['ACLED_Month'] = df['TimeFK_Event_Date'].apply(lambda x: str(x)[4:6])

# COMMAND ----------

df.columns

# COMMAND ----------

df.groupby('ACLED_Event_Type').size()

# COMMAND ----------

df.groupby('ACLED_Event_Type').agg({'ACLED_Fatalities':np.mean})

# COMMAND ----------

df.groupby(['ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Year','ACLED_Month']).agg({'ACLED_Fatalities':np.sum}).reset_index()['ACLED_Fatalities'].sort_values().iloc[:1260]

# COMMAND ----------

df.groupby('Fatal').size()

# COMMAND ----------

df.groupby('ACLED_Event_Type').size()

# COMMAND ----------



# COMMAND ----------


