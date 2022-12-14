# Databricks notebook source
# MAGIC %md # ML from Disaster: Sinking of the Titanic
# MAGIC 
# MAGIC This simple example will build a feature store on top of data from the Titanic passenger list and then use it to train a model and deploy both the model and features to production.
# MAGIC </br>
# MAGIC </br>
# MAGIC 
# MAGIC <img src = "https://cdn.britannica.com/79/4679-050-BC127236/Titanic.jpg" width=700>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ##### Background
# MAGIC - The sinking of the Titanic is one of the most infamous shipwrecks in history.
# MAGIC - On April 15, 1912, during her maiden voyage, the widely considered "unsinkable" RMS Titanic sank after colliding with an iceberg. Unfortunately there weren't enough lifeboards of everyone onboard resulting in the death of 1502 of 2224 passengers and crew.
# MAGIC - While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC __The goal is to produce a predictive model that is able to predict which passengers survived using passenger data (ie. name, age, gender, socia-economic class ect.)__

# COMMAND ----------

# MAGIC %md ## 1. Perform Initial Setup
# MAGIC 
# MAGIC Let's first create a database to hold the data assets for this example, and set that database as the default for this notebook.

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS titanic

# COMMAND ----------

# MAGIC %sql USE titanic

# COMMAND ----------

# MAGIC %md We can use the SparkFiles API to download data from a URL and load into a DataFrame.

# COMMAND ----------

from pyspark import SparkFiles

test_url = "https://raw.githubusercontent.com/guanjieshen/nait_2022/master/titanic_test.csv"
train_url = "https://raw.githubusercontent.com/guanjieshen/nait_2022/master/titanic_train.csv"

sc.addFile(test_url)
sc.addFile(train_url)

test_path = SparkFiles.get('titanic_test.csv')
train_path = SparkFiles.get('titanic_train.csv')

df_train = spark.read.csv("file://" + train_path, header="True", inferSchema="True")
df_test = spark.read.csv("file://" + test_path, header="True", inferSchema="True")

# COMMAND ----------

# MAGIC %md Let's take a look at the data set, and run a data profile to understand the features available.

# COMMAND ----------

display(df_train)

# COMMAND ----------

# MAGIC %md ## 2. Cleaning the Data Set
# MAGIC 
# MAGIC Let's try and create a bit more clarity around the column names within the intial data set.

# COMMAND ----------

titanic_train = (df_train
                 .withColumnRenamed("Pclass", "PassengerClass")
                 .withColumnRenamed("SibSp", "SiblingsSpouses")
                 .withColumnRenamed("Parch", "ParentsChildren")
                )
                 
titanic_test = (df_test
                 .withColumnRenamed("Pclass", "PassengerClass")
                 .withColumnRenamed("SibSp", "SiblingsSpouses")
                 .withColumnRenamed("Parch", "ParentsChildren")
                )
                 

# COMMAND ----------

titanic_train.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("titanic.passengers_train")

titanic_test.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("titanic.passengers_test")

