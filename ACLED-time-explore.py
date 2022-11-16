# Databricks notebook source
import pandas as pd
import numpy as np
from db_keys import keys

# COMMAND ----------

def get_data(keys, cnty_code, year_lst):
    database_host = keys["database_host"]
    database_port = keys["database_port"]
    database_name = keys["database_name"]
    user = keys["user"]
    password = keys["password"]
    
    table = "dbo.CRD_ACLED"
    url = f"jdbc:sqlserver://{database_host}:{database_port};databaseName={database_name};"

    df = (spark.read
      .format("com.microsoft.sqlserver.jdbc.spark")
      .option("url", url)
      .option("dbtable", table)
      .option("user", user)
      .option("password", password)
      .load()
    )
    
    # sudan country code
    df = table.filter(df.CountryFK==cnty_code) 
    df = df.toPandas()
    # create year-month column
    df['YearMonth'] = df['TimeFK_Event_Date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6])    
    # subset to particular years
    df_sub = df[df['ACLED_Year'].isin(year_lst)]
    print(f'The full dataset has {df.shape[0]} rows.')
    print(f'The year subset dataset has {df_sub.shape[0]} rows.')
    
    return df, df_sub

# COMMAND ----------

df, df_sub = get_data(keys, ['2020','2021','2022'])

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

counts = make_lagged_features(df_sub, 3, 'YearMonth', 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_PK', 'count')

# COMMAND ----------

fatal = make_lagged_features(df_sub, 3, 'YearMonth', 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', 'sum')

# COMMAND ----------

# df[(df.YearMonth=='2020-07') & (df.ACLED_Admin1=='West Darfur') & (df.ACLED_Event_Type=='Battles')]['ACLED_Fatalities'].sum()

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


def get_months_since_df(data, m_since_lst, date_col, admin_col, event_type_col, value_col, start_year_month):
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
    idx = list(months_since_data.index).index(start_year_month) #'2020-04'
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

mon_dat = get_months_since_df(df, [1, 5, 50], 'YearMonth', 'ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Fatalities', '2020-04')

# COMMAND ----------

X = pd.concat([counts, fatal, mon_dat], axis=1)
X.fillna(0, inplace=True)

# COMMAND ----------

y = df_sub.groupby(['YearMonth', 'ACLED_Admin1']).agg({'ACLED_Fatalities': 'sum'})
y = pd.concat([X, y], axis=1)['ACLED_Fatalities']
y.fillna(0, inplace=True)

# COMMAND ----------

idy = [i for i in y.index if i[0] not in ['2020-01', '2020-02', '2020-03']]

# COMMAND ----------

y = y.loc[idy]

# COMMAND ----------

X.to_csv('/dbfs/FileStore/df/sudan.csv')

# COMMAND ----------

X_train = X.iloc[:506, :]
X_test = X.iloc[506:, :]

y_train = y.iloc[:506]
y_test = y.iloc[506:]

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns

# COMMAND ----------

!pip install scikit-optimize
!pip install xgboost

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from skopt import BayesSearchCV 
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Modeling function
params={'booster':['gblinear'],
        'objective': ['reg:squarederror'],
        'min_child_weight': (0, 50,),
        'max_depth': (0, 10),
        'colsample_bytree': (0.5, 1.0),
        'lambda':(0.00001,10),
        'alpha':(0.00001,10),
        'learning-rate':(0.01,0.2,'log-uniform')
        }

def fit_model(X_train, y_train, params, num_tries):
    # model
    bayes = BayesSearchCV(
        xgb.XGBRegressor(),
        params,
        n_iter=num_tries,
        cv=5,
        random_state=42)

    fitted_model = bayes.fit(X_train, y_train)
    pred = fitted_model.predict(X_test)

    # results
    res = pd.DataFrame(fitted_model.cv_results_)
    return fitted_model, pred, res

# COMMAND ----------

model, predictions, results = fit_model(X_train, y_train, params, 50)

# COMMAND ----------

predictions

# COMMAND ----------

# print("CV-test RMSE:", np.sqrt(model.best_score_))
print("pred RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("pred R2:", r2_score(y_test, predictions))

# COMMAND ----------

import matplotlib.pyplot as plt
x = ['2022-03',
 '2022-04',
 '2022-05',
 '2022-06',
 '2022-07',
 '2022-08',
 '2022-09',
 '2022-10']
for admin in df.ACLED_Admin1.unique():
    actual = []
    pred = []
    mx = [x for x in p.index if x[1]==admin]
    for m in mx:
        actual.append(p.loc[m, 'actual'])
        pred.append(p.loc[m, 'pred'])

    plt.plot(x, actual)
    plt.plot(x, pred, '-.')
    plt.title(admin)
    plt.show()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

df.groupby(['ACLED_Source']).agg({'ACLED_Fatalities':[np.mean, np.sum]}).sort_values(('ACLED_Fatalities','mean'), ascending=False)

# COMMAND ----------

df['ACLED_Month'] = df['TimeFK_Event_Date'].apply(lambda x: str(x)[4:6])

# COMMAND ----------

df.columns

# COMMAND ----------

df.groupby('ACLED_Event_Type').size()

# COMMAND ----------

df.groupby('ACLED_Event_Type').agg({'ACLED_Fatalities':np.mean})

# COMMAND ----------

pd.set_option('display.max_rows', 1000)

# COMMAND ----------

df.groupby(['ACLED_Admin1', 'ACLED_Event_Type', 'ACLED_Year','ACLED_Month']).agg({'ACLED_Fatalities':np.sum}).reset_index()['ACLED_Fatalities'].sort_values().iloc[:1260]

# COMMAND ----------

df.groupby('Fatal').size()

# COMMAND ----------

df['Fatal'] = df['ACLED_Fatalities'].apply(lambda x: str(x))

# COMMAND ----------

df.head(3)

# COMMAND ----------

df.groupby('ACLED_Event_Type').size()

# COMMAND ----------

from datetime import datetime

# COMMAND ----------

def convert_dt(value):
    valstr = str(value)
    date_clean = datetime(year=int(valstr[0:4]), month=int(valstr[4:6]), day=int(valstr[6:8]))
    return date_clean

conflict_ts = df.copy()
conflict_ts['Date_Clean'] = conflict_ts['TimeFK_Event_Date'].apply(lambda x: convert_dt(x))
conflict_ts[['Date_Clean','TimeFK_Event_Date']]

# Filter by year
conflict_ts = conflict_ts[conflict_ts['Date_Clean'].dt.year >= 2019]

# Aggregate
conflict_ts = conflict_ts.groupby(['ACLED_Admin2', 'Date_Clean']).agg({'ACLED_Admin2':'count'})
conflict_ts.rename(columns={'ACLED_Admin2':'Event Count'}, inplace=True)
conflict_ts.reset_index(inplace=True)

# Some vars to make generation of Cartesion Index easy
mindate = min(conflict_ts['Date_Clean'])
maxdate = max(conflict_ts['Date_Clean'])
z = conflict_ts['ACLED_Admin2'].unique()

# Set multi-time index
conflict_ts.set_index(['ACLED_Admin2','Date_Clean'], inplace=True)


# COMMAND ----------

import pandas as pd

# COMMAND ----------

# Gen Dataframe with cartesian index
days = pd.date_range(start=mindate,end=maxdate)
new_index = pd.MultiIndex.from_product([z,days])
new_df = conflict_ts.reindex(new_index)

# Check initial results
new_df['Event Count'].sum() # lose 5 events due to 5 admin 2 nas

# Fill missing values with zeros 
# Key assumption: We attach a zero to instances where there is a missing Admin/date combo assuming no conflict event occured (e.g. there is no acled events fo that date/admin).
new_df['Event Count'].fillna(0, inplace=True)

# COMMAND ----------

### Time Series Feature Engineering
##Time Series Aggregation
# Create copy
tsagg = new_df.copy()

# Agg data into event counts based ons specified period
# Default freq code is byweekly - see list of options here - https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
tsagg = pd.DataFrame(tsagg.groupby(level=0).resample("SMS", level=1)['Event Count'].sum())
tsagg.index.set_names(['ACLED_Admin', 'Date'], inplace=True)

# COMMAND ----------

tsagg.index

# COMMAND ----------

tsagg

# COMMAND ----------

!pip install tsfresh

# COMMAND ----------

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series, make_forecasting_frame

# COMMAND ----------

### Time Series Feature Engineering
## Feature Generation
# Set extraction parameters
extraction_settings = ComprehensiveFCParameters()


# Don't need multime index for tsfresh
tsfresh_df = tsagg.reset_index()

# Create rolling windows
df_rolled = roll_time_series(tsfresh_df, column_id="ACLED_Admin", column_sort="Date",
                             max_timeshift=12, min_timeshift=1)


# Extract Features
X = extract_features(df_rolled.drop('ACLED_Admin', axis=1),
                     column_id='id', column_sort="Date", column_value="Event Count", 
                     impute_function=impute, show_warnings=True)

# Ensure alignment
# make sure y variable matches x features in terms of time period
y = pd.DataFrame(tsagg.sort_index().groupby(['ACLED_Admin'])['Event Count'].shift(-1))

# COMMAND ----------

X

# COMMAND ----------

y

# COMMAND ----------

y['test'] = y.isna()
print(y['test'].value_counts()) # 25
# adms
print(len(tsagg.index.get_level_values(0).unique())) # 25

# Secondary shift check - shift consistency
print(y.head())
tsagg[:6]

# drop nas and test variable
y.dropna(inplace=True)
y.drop('test', axis=1, inplace=True)

# Ensure indexes x features align
y = y[y.index.isin(X.index)]
X = X[X.index.isin(y.index)]

# COMMAND ----------

def train_test_split(data,yvar, split=0.80):
    size=int(len(data)*split)
    # for train data will be collected from each country's data which index is from 0-size (80%)
    x_train =data.sort_index().iloc[0:size] 
    # for test data will be collected from each country's  data which index is from size to the end (20%)
    x_test =data.sort_index().iloc[size:]
    y_train=yvar.iloc[0:size] 
    y_test=yvar.iloc[size:] 
    return x_train, x_test,y_train,y_test



# loop each country_Region and split the data into train and test data
X_train=[]
X_test=[]
Y_train=[]
Y_test=[]

# Function sort by admin so as not to compromise accuracy results /
# (e.g. train on x percent of data for each admin and predict on the other 20 so as not to inflate model accuracy)
admins = list(set(pd.Series(X.index.get_level_values(0))))


# COMMAND ----------

X_Filt = X

# COMMAND ----------

for i in range(0,len(admins)):
    df = X_Filt.loc[(X_Filt.index.get_level_values(0) == admins[i])] # Need to make this flexible to admin size for now change manually
    print(df.head())
    ydf = y.loc[(y.index.get_level_values(0) == admins[i])]
    print(ydf.head())
    x_train, x_test,y_train,y_test=train_test_split(df, ydf['Event Count'])
    X_train.append(x_train)
    X_test.append(x_test)
    Y_train.append(y_train)
    Y_test.append(y_test)


Y_test = pd.concat(Y_test)
Y_train = pd.concat(Y_train)
X_train = pd.concat(X_train)
X_test = pd.concat(X_test)

# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# COMMAND ----------

n_iter = 10
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)
random_state = 59
rf_model = RandomForestRegressor(random_state=random_state)
rf_params = {
    "max_depth": [2,6,12],
    "n_estimators": [10, 25, 40]
}
cv_obj = RandomizedSearchCV(
    rf_model,
    param_distributions=rf_params,
    n_iter=n_iter,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    random_state=random_state,
    verbose=0,
    n_jobs=-1,
)
cv_obj.fit(X_train, Y_train)
best_est = cv_obj.best_estimator_

# Estimate prediction intervals on test set with best estimator
# Here, a non-nested CV approach is used for the sake of computational
# time, but a nested CV approach is preferred.
# See the dedicated example in the gallery for more information.
alpha = 0.3

# COMMAND ----------

cv_obj.cv_results_

# COMMAND ----------

y_pred = cv_obj.predict(X_test)

# COMMAND ----------

import matplotlib.pyplot as plt
# Plot estimated prediction intervals on test set
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel("Hourly demand (GW)")
ax.plot(Y_test.values, lw=2, label="Test data", c="C1")
ax.plot(
    y_pred,
    lw=2,
    c="C2",
    label="Predictions"
)
ax.fill_between(
    Y_test.index,
    y_pis[:, 0, 0],
    y_pis[:, 1, 0],
    color="C2",
    alpha=0.2,
    label="CV+ PIs"
)
ax.legend()
plt.show()

# COMMAND ----------



# COMMAND ----------

df['datetime'] = df['TimeFK_Event_Date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))

# COMMAND ----------

times = pd.date_range('1998-10-01', periods=1254, freq='1w')

# COMMAND ----------

series = []
for i, t in enumerate(times):
    d = df[(df['datetime']>times[i]) & (df['datetime']<times[i+1])]
    if d.shape[0] > 0:
        sm = d.ACLED_Fatalities.sum()
    else:
        sm = 0
    series.append(sm)

# COMMAND ----------

series

# COMMAND ----------

d = df[(df['datetime']>times[3]) & (df['datetime']<times[4])]
d.ACLED_Fatalities.sum()

# COMMAND ----------

times

# COMMAND ----------


