# Databricks notebook source
# MAGIC %md
# MAGIC **What**: This notebook uses `MLFlow` and `hyperopt` to train DL models. 
# MAGIC
# MAGIC **How**: Review and change things under CHANGE ME before running.   
# MAGIC   
# MAGIC **Things that still need work**  
# MAGIC     * Figure out how to create sample weights to counter the imbalance and implement (see classic ML notebook)  
# MAGIC     * Review `objective` function and do some testing  
# MAGIC     * Consider adding more hyperparameters (see TODO list cell with hyperparam space)

# COMMAND ----------

# MAGIC %md
# MAGIC ## CHANGE ME

# COMMAND ----------

import mlflow
import databricks.automl_runtime

# CREATE MLFLOW EXPERIMENT FIRST
EXP_ID = "3871664251815056" 

# COMMAND ----------

# designate here
database_name = "horn_africa_forecast"
data_table = ""
target_col = "binary_escalation_30"
time_col = "STARTDATE"

# number of embeddings in each set of lags (full is 512, can be less if using PCA data)
N_EMBED = 50
# number of lags
N_LAGS = 4
# number of hyperparameter sets to try
MAX_EVALS = 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# read in data
spdf = spark.sql(f"SELECT * FROM {database_name}.{data_table}")
df = spdf.toPandas()

# COMMAND ----------

### Split data
# train-val-test 60-20-20 on date column
import math

# split date
all_time = df[time_col].unique()
all_time.sort()
train_end = math.ceil(len(all_time) * 0.6)
val_end = math.ceil(len(all_time) * 0.8)
train_dt = all_time[ :train_end]
val_dt = all_time[train_end:val_end]
test_dt = all_time[val_end: ]

# create col for splitting
df['_automl_split_col_0000'] = ''
df.loc[df[time_col].isin(train_dt), '_automl_split_col_0000'] = 'train'
df.loc[df[time_col].isin(val_dt), '_automl_split_col_0000'] = 'val'
df.loc[df[time_col].isin(test_dt), '_automl_split_col_0000'] = 'test'

# COMMAND ----------

### Create sample weights to counter the imbalance
# hardcoding for now -- CHANGE ME LATER!

df['_automl_sample_weight_0000'] = 1
df.loc[(df['_automl_split_col_0000']=='train') & (df[target_col]==0), '_automl_sample_weight_0000'] = 1.4838198687485855

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
import numpy as np
import pandas as pd

conf_hist = ['Battles', 'Explosions_Remote_violence', 'Protests', 'Riots', 'Strategic_developments', 'Violence_against_civilians']
embed = np.arange(N_EMBED)
static = ['female_0_14_2020','male_0_14_2020','female_15_64_2020','male_15_64_2020','female_65_plus_2020','male_65_plus_2020', 'conflict_trend_1', 'conflict_trend_2']

# create t-x column names
conf_hist = [f'{col}_t-{x}' for col in conf_hist for x in np.arange(1, N_LAGS+1)]
embed = [f'{col}_t-{x}' for col in embed for x in np.arange(1, N_LAGS+1)]
# column selector
supported_cols = conf_hist + embed + static

col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# take out unneeded columns
keep_cols = supported_cols + [target_col, '_automl_sample_weight_0000', '_automl_split_col_0000']
df = df[keep_cols].copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), supported_cols)) # not really needed, but doing it just in case

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, supported_cols)]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers
preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC
# MAGIC `_automl_split_col_0000` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.
# MAGIC
# MAGIC Given that `STARTDATE` is provided as the `time_col`, the data is split based on time order,
# MAGIC where the most recent data is split to the test data.

# COMMAND ----------

df_loaded = df
# AutoML completed train - validation - test split internally and used _automl_split_col_0000 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "val"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "test"]

# Separate target column from features and drop _automl_split_col_0000
X_train = split_train_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_test = split_test_df[target_col]

# COMMAND ----------

# AutoML balanced the data internally and use _automl_sample_weight_0000 to calibrate the probability distribution
sample_weight = X_train.loc[:, "_automl_sample_weight_0000"].to_numpy()
X_train = X_train.drop(["_automl_sample_weight_0000"], axis=1)
X_val = X_val.drop(["_automl_sample_weight_0000"], axis=1)
X_test = X_test.drop(["_automl_sample_weight_0000"], axis=1)

# input dimension for first NN layer
INPUT_DIM = X_train.shape[1]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/4171730171948156)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

!pip install scikeras

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.keras
import mlflow.tensorflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow import pyfunc

from hyperopt import hp, tpe, fmin, STATUS_OK, SparkTrials

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
mlflow.sklearn.autolog(disable=True)
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

# COMMAND ----------


# model builder
def create_model(layer_choice, units0, units1, dropout1, activation, theoptimizer):
    model = Sequential()
    # input layer
    model.add(Dense(int(units0), input_dim=INPUT_DIM, activation=activation))
    # hidden layers #
    model.add(Dense(int(units1), activation=activation))
    model.add(Dropout(dropout1))
    if layer_choice['layers'] == 'two':
        model.add(Dense(int(layer_choice['units2']), activation=activation))
        model.add(Dropout(layer_choice['dropout2']))
    elif layer_choice['layers'] == 'three':
        model.add(Dense(int(layer_choice['units2_']), activation=activation))
        model.add(Dropout(layer_choice['dropout2_']))
        model.add(Dense(int(layer_choice['units3']), activation=activation))
        model.add(Dropout(layer_choice['dropout3']))
    # output layer
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=theoptimizer, metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model


def objective(params):
    with mlflow.start_run(experiment_id=EXP_ID) as mlflow_run:
        # classifier
        clf = KerasClassifier(build_fn=create_model, layer_choice=params['choice'], units0=params['units0'], units1=params['units1'], dropout1=params['dropout1'], activation=params['activation'], theoptimizer=params['opt'])
        # build pipeline
        model = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(
            log_input_examples=True,
            silent=True)

        # fit the model
        model.fit(X_train, y_train, classifier__batch_size=params['batch_size'], classifier__epochs=100, classifier__callbacks=EarlyStopping(patience=10, monitor="val_loss"), classifier__validation_data=(X_val_processed, y_val), classifier__sample_weight=sample_weight)

        # Log metrics for the training set
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = pyfunc.PyFuncModel(model_meta=mlflow_model, model_impl=model)
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(target_col):y_train}),
            targets=target_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "training_" , "pos_label": 1, "sample_weight": sample_weight }
        )
        training_metrics = training_eval_result.metrics
        # Log metrics for the validation set
        val_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_val.assign(**{str(target_col):y_val}),
            targets=target_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "val_" , "pos_label": 1 }
        )
        val_metrics = val_eval_result.metrics
        # Log metrics for the test set
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(target_col):y_test}),
            targets=target_col,
            model_type="classifier",
            evaluator_config = {"log_model_explainability": False,
                                "metric_prefix": "test_" , "pos_label": 1 }
        )
        test_metrics = test_eval_result.metrics

        loss = -val_metrics["val_f1_score"]

        # Truncate metric key names so they can be displayed together
        val_metrics = {k.replace("val_", ""): v for k, v in val_metrics.items()}
        test_metrics = {k.replace("test_", ""): v for k, v in test_metrics.items()}

        return {
        "loss": loss,
        "status": STATUS_OK,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model": model,
        "run": mlflow_run,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree classifier, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

"""
TODO
Model.compile: https://keras.io/api/models/model_training_apis/
Optimizers: https://keras.io/api/optimizers/
    optimizer, learning_rate, momentum
Model.fit: https://keras.io/api/models/model_training_apis/
    batch_size, epochs, sample_weight
Consider also:
    Dropout, L1/L2, more or less Dense layers
See also: https://keras.io/guides/keras_tuner/getting_started/
"""

space = {'choice': hp.choice('num_layers',
                    [
                        {'layers':'one'},
                        {'layers':'two',
                         'units2': hp.choice('units2', [126, 256, 512, 1024]),
                         'dropout2': hp.uniform('dropout2', .25, .5)
                         },
                        {'layers':'three',
                         'units2_': hp.choice('units2_', [126, 256, 512, 1024]),
                         'dropout2_': hp.uniform('dropout2_', .25, .5),
                         'units3': hp.choice('units3', [126, 256, 512, 1024]),
                         'dropout3': hp.uniform('dropout3', .25, .5)
                         }
                    ]),
         'units0': hp.choice('units0', [126, 256, 512, 1024]),
         'units1': hp.choice('units1', [126, 256, 512, 1024]),
         'dropout1': hp.uniform('dropout1', .25, .5),
         'activation': hp.choice('activation', ['relu', 'tanh']),
         'opt': hp.choice('optimizer', ['adam', 'rmsprop']),
         'batch_size': hp.choice('batch_size', [16, 32, 64])
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

# run trials
trials = SparkTrials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=MAX_EVALS,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
model

# COMMAND ----------

idx_dict = trials.best_trial['misc']['vals']
idx_dict

# COMMAND ----------

# MAGIC %md
# MAGIC ### Skipping model registration -- see auto generated notebooks for this code

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC and Precision-Recall curves for validation data
# MAGIC
# MAGIC We show the confusion matrix, ROC and Precision-Recall curves of the model on the validation data.
# MAGIC
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Click the link to see the MLflow run page
displayHTML(f"<a href=#mlflow/experiments/{ EXP_ID }/runs/{ mlflow_run.info.run_id }/artifactPath/model> Link to model run page </a>")

# COMMAND ----------

import os
import uuid
from IPython.display import Image

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve_plot.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve_plot.png")
display(Image(filename=eval_pr_curve_path))

# COMMAND ----------


