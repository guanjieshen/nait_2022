# Databricks notebook source
# MAGIC %md ## Make Predictions Using the Moddel

# COMMAND ----------

titanic_validate =spark.read \
  .table("titanic.passengers_test_features")

display(titanic_validate)

# COMMAND ----------

from databricks import feature_store

model_name = "titanic_lr_enb"
stage = "Staging"
model_uri=f"models:/{model_name}/{stage}"

fs = feature_store.FeatureStoreClient()

predictions = fs.score_batch(
    model_uri,
    titanic_validate
)

display(predictions)

# COMMAND ----------


