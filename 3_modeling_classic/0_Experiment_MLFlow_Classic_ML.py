# Databricks notebook source
# MAGIC %md
# MAGIC **What**: This notebook uses `MLFlow` and `hyperopt` to train classic ML models. 
# MAGIC
# MAGIC **How**: Review and change things under CHANGE ME before running.   
# MAGIC   
# MAGIC **Things that still need work**  
# MAGIC     * Figure out how to create sample weights to counter the imbalance and implement  
# MAGIC     * This notebook only has lightGBM -- add in other algos (XGBoost, etc.) to have everything in one notebook  
# MAGIC     * Review `objective` function and do some testing before running a real experiment

# COMMAND ----------

# MAGIC %md
# MAGIC ## CHANGE ME

# COMMAND ----------

import mlflow
import databricks.automl_runtime

# CREATE MLFLOW EXPERIMENT FIRST
EXP_ID = "3050380669319955"

# COMMAND ----------

# designate here
database_name = "horn_africa_forecast_base"
data_table = "cameo1_titlefill_sumfat_1w_popdense_conftrend_mod"
target_col = "bin_esc_30"
time_col = "STARTDATE"

# number of embeddings in each set of lags (full is 512, can be less if using PCA data)
N_EMBED = 512
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/4171730171948156)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

import lightgbm
from lightgbm import LGBMClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

def objective(params):
  with mlflow.start_run(experiment_id=EXP_ID) as mlflow_run:
    lgbmc_classifier = LGBMClassifier(**params)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("classifier", lgbmc_classifier),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True)

    model.fit(X_train, y_train, classifier__sample_weight=sample_weight)

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(target_col):y_train}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": 1, "sample_weight": sample_weight }
    )
    sklr_training_metrics = training_eval_result.metrics
    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(target_col):y_val}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "val_" , "pos_label": 1 }
    )
    sklr_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(target_col):y_test}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": 1 }
    )
    sklr_test_metrics = test_eval_result.metrics

    loss = -sklr_val_metrics["val_f1_score"]

    # Truncate metric key names so they can be displayed together
    sklr_val_metrics = {k.replace("val_", ""): v for k, v in sklr_val_metrics.items()}
    sklr_test_metrics = {k.replace("test_", ""): v for k, v in sklr_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": sklr_val_metrics,
      "test_metrics": sklr_test_metrics,
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
# MAGIC https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

# read about each param of LightGBM here https://lightgbm.readthedocs.io/en/latest/Parameters.html

space = {
    "colsample_bytree": hp.uniform('colsample_bytree', 0, 1), # colsample_bytree between 0 and 1
    "lambda_l1": hp.choice('lambda_l1', range(0, 12)), # lambda_l1 >= 0.0
    "lambda_l2": hp.choice('lambda_l2', range(0, 12)), # lambda_l2 >= 0.0
    "learning_rate": hp.uniform('learning_rate', 0.01, 4), 
    "max_bin": hp.choice('max_bin', range(1, 500, 50)), # max_bin > 1
    "max_depth": 0, # this is useful when data is small, not our case 
    "min_child_samples": hp.choice('min_child_samples', [20, 50, 75, 100, 125, 150, 200, 300, 500]),
    "n_estimators": hp.choice('n_estimators', [20, 50, 100, 200, 500]), # Default is num_class * num_iterations
    "num_leaves": hp.choice('num_leaves', [1, 30, 50, 100, 1000, 50000, 10000]), # 1 < num_leaves <= 131072
    "path_smooth": hp.choice('path_smooth', [1, 10, 25, 50, 100, 150]),
    "subsample": hp.uniform('subsample', 0, 1), # between 0 and 1
    "random_state": hp.choice('random_state', [2349874, 277234093,])}

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

from hyperopt import SparkTrials
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


