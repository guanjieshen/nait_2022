# Databricks notebook source
# MAGIC %md ### Train the Model
# MAGIC 
# MAGIC Train a LightGBM on the data, then log the model with MLFlow.

# COMMAND ----------

titanic_train = spark.read.table("titanic.passengers_train_features")
display(titanic_train)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
import lightgbm as lgb
import mlflow
from sklearn.metrics import accuracy_score
import pandas as pd
import re

data = titanic_train.toPandas()

# Convert Categorical Variables into dummy/indicator variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum = data_dum.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

display(data_dum)

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Extract features & labels
x = data_dum.drop(["Survived"], axis=1)
y = data_dum.Survived

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lgb_params = {
  'n_estimators': 75,
  'learning_rate': 1e-3,
  'subsample': 0.2767,
  'colsample_bytree': 0.6,
  'reg_lambda': 1e-1,
  'num_leaves': 50,
  'max_depth':8
}

lgbm_clf = lgb.LGBMClassifier(**lgb_params)

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

with mlflow.start_run(run_name="lightgbm") as mlflow_run:
  lgbm_clf.fit(x_train, y_train)
      # Log metrics for the test set
  lgbmc_test_metrics = mlflow.sklearn.eval_and_log_metrics(lgbm_clf, x_test, y_test, prefix="test_")
  lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}
  display(pd.DataFrame([lgbmc_test_metrics], index=["test"]))


# COMMAND ----------

lgb_pred = lgbm_clf.predict(x_test)

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, lgbm_clf, result_type="double")
output_df = spark.table("passengers_test_features").withColumn("prediction", predict(struct(*table.columns)))

# COMMAND ----------


