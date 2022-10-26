# Databricks notebook source
# MAGIC %md ##Feature Engineering
# MAGIC 
# MAGIC At this point the we want to apply additional feature engineering to create a denormalized dataset in order to do model development.

# COMMAND ----------

titanic_passengers = spark.read.format('delta').table("titanic.passengers_train")

# COMMAND ----------

# MAGIC %md #### Passenger Title

# COMMAND ----------

display(titanic_passengers.select("Name"))

# COMMAND ----------

# MAGIC %md If we are able to extract titles from the names, we can get more information on social status, profession ect.
# MAGIC 
# MAGIC Let's do the folowing:
# MAGIC - Extract Title from Name, and store in column "Title"
# MAGIC - Use regex parsing on `Name` column to create a new feature/column `Title`

# COMMAND ----------

from pyspark.sql.functions import *
name_parse_df = titanic_passengers.withColumn("Title", regexp_extract(col("Name"),"([A-Za-z]+)\.",1))

display(name_parse_df.select("Title","Name"))

# COMMAND ----------

# MAGIC %md From this new DataFrame, let's see if we can sanitize the titles our even further:
# MAGIC ```
# MAGIC 'Mlle', 'Mne', 'Ms' --> Miss
# MAGIC 'Lady', 'Dona', 'Countess' --> Mrs
# MAGIC 'Dr', 'Master', 'Major', 'Capt', 'Sir', 'Don' --> Mr
# MAGIC 'Jonkheer', 'Col', 'Rev' --> Other
# MAGIC ```

# COMMAND ----------

miss_replace = {'Mlle': 'Miss', 'Mme': 'Miss','Ms': 'Miss'}
mrs_replace = {'Lady': 'Mrs', 'Dona': 'Mrs','Countess': 'Mrs'}
mr_replace = {'Dr': 'Mr', 'Master': 'Mr','Major': 'Mr','Capt': 'Mr','Sir': 'Mr','Don': 'Mr'}
other_replace = {'Jonkheer': 'Other', 'Col': 'Other','Rev': 'Other'}

name_normalize_df = name_parse_df.replace(miss_replace).replace(mrs_replace).replace(mr_replace).replace(other_replace)

display(name_normalize_df)

# COMMAND ----------

# MAGIC %md #### Passenger's Cabins

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's extract to see if a passenger had a dedicated cabin on the ship.

# COMMAND ----------

has_cabin_df = name_normalize_df.withColumn("HasCabin", name_normalize_df.Cabin.isNotNull())
display(has_cabin_df)

# COMMAND ----------

# MAGIC %md #### Family Size
# MAGIC 
# MAGIC Let's see if the total size of the family will be important in terms of overall survivability. 
# MAGIC 
# MAGIC In this case this would be the number of siblings & spouse plus the number of parents & children.

# COMMAND ----------

family_size_df = has_cabin_df.withColumn("FamilySize", col("SiblingsSpouses") + col("ParentsChildren"))
display(family_size_df)

# COMMAND ----------

# MAGIC %md Let's organize some of the newly created features into a new Dataframe

# COMMAND ----------

titanic_features = family_size_df

titanic_features.write \
  .format("delta") \
  .mode("overwrite") \
  .saveAsTable("titanic.passenger_train_features")


# COMMAND ----------

# MAGIC %md ### Let's turn the code into a function and apply it to the test data set.

# COMMAND ----------

from pyspark.sql.functions import *


def addAdditionalFeatures(input_table, output_table):
  # Normalize column names
  input_df = (spark.read.table(input_table)
                 .withColumnRenamed("Pclass", "PassengerClass")
                 .withColumnRenamed("SibSp", "SiblingsSpouses")
                 .withColumnRenamed("Parch", "ParentsChildren")
                )
  # Add title feature
  name_parse_df = input_df.withColumn("Title", regexp_extract(col("Name"),"([A-Za-z]+)\.",1))
                       
  # Normalize titles
  miss_replace = {'Mlle': 'Miss', 'Mme': 'Miss','Ms': 'Miss'}
  mrs_replace = {'Lady': 'Mrs', 'Dona': 'Mrs','Countess': 'Mrs'}
  mr_replace = {'Dr': 'Mr', 'Master': 'Mr','Major': 'Mr','Capt': 'Mr','Sir': 'Mr','Don': 'Mr'}
  other_replace = {'Jonkheer': 'Other', 'Col': 'Other','Rev': 'Other'}
  name_normalize_df = name_parse_df.replace(miss_replace).replace(mrs_replace).replace(mr_replace).replace(other_replace)
                       
  # Add has cabin feature
  has_cabin_df = name_normalize_df.withColumn("HasCabin", name_normalize_df.Cabin.isNotNull())
                       
  # Add family size feature
  family_size_df = has_cabin_df.withColumn("FamilySize", col("SiblingsSpouses") + col("ParentsChildren"))
  
  # Write to Delta Table
  family_size_df.write.format("delta").saveAsTable(output_table)
   

# COMMAND ----------

addAdditionalFeatures("titanic.passengers_test","titanic.passengers_test_features")
