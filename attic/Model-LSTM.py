# Databricks notebook source
import tensorflow as tf
import keras

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

# COMMAND ----------

import re
from math import sqrt, ceil
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# COMMAND ----------

# designate the outcome var and training dataset proportion
OUTCOME = 'binary_escalation_30'
train_pct = .7

# COMMAND ----------

# read-in dataset and convert to pandas
dataset = spark.sql('SELECT * FROM news_media.horn_africa_model_escbin_emb_confhist_lag_m4_gld')
dataset = dataset.toPandas()

# COMMAND ----------

# sorting by date is important for time series!
dataset = dataset.sort_values(['STARTDATE','COUNTRY','ADMIN1'])

# COMMAND ----------

# make sure t-x features in the same order across all the t-x's
all_cols = dataset.columns
t1 = [x for x in all_cols if re.search('_t-1', x)]
t1.sort()
t2 = [x for x in all_cols if re.search('_t-2', x)]
t2.sort()
t3 = [x for x in all_cols if re.search('_t-3', x)]
t3.sort()
t4 = [x for x in all_cols if re.search('_t-4', x)]
t4.sort()

# COMMAND ----------

# figure out where to split the data by date - get the index
dates = dataset.STARTDATE.unique()
dates.sort()
next_date = dates[ceil(len(dates) * train_pct)]
next_date_i = dataset[dataset.STARTDATE==next_date].first_valid_index()

# COMMAND ----------

# just features and outcome in the right order
dataset = dataset.loc[:, [*t1, *t2, *t3, *t4, OUTCOME]]

# COMMAND ----------

# split features and outcome
dataset, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

# COMMAND ----------

dataset = dataset.values
# ensure all data is float
dataset = dataset.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# COMMAND ----------

# specify the number of lags and features
n_tmins = 4
n_features = 518

# COMMAND ----------

# split into input and outputs
train_X = dataset[:next_date_i, :]
train_y = y[:next_date_i]
test_X = dataset[next_date_i:, :]
test_y = y[next_date_i:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_tmins, n_features))
test_X = test_X.reshape((test_X.shape[0], n_tmins, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# COMMAND ----------

del dataset

# COMMAND ----------

#https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# COMMAND ----------

# design network
tf.keras.backend.clear_session()
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m])
# fit network
history = model.fit(train_X, train_y, epochs=60, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# COMMAND ----------

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# COMMAND ----------

loss, accuracy, f1_score, precision, recall = model.evaluate(test_X, test_y, verbose=0)

# COMMAND ----------

accuracy, f1_score, precision, recall

# COMMAND ----------


