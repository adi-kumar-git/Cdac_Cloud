#!/usr/bin/env python
# coding: utf-8

# In[83]:


get_ipython().system('pip install pyspark')


# In[ ]:





# In[84]:


from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder     .appName("PySpark Example")     .getOrCreate()

# Print Spark version to verify it's working
print("Spark Version:", spark.version)


# In[85]:


from pyspark.sql import SparkSession
spark= SparkSession.builder     .appName("PySpark Example")     .getOrCreate()
print(spark.version)


# In[86]:


# Create a DataFrame from a List
data = [("John", 28), ("Anna", 23), ("Mike", 32)]
columns = ["Name", "Age"]


# In[87]:


df = spark.createDataFrame(data,columns)
df.show()


# In[ ]:





# In[88]:


print(dir(spark))


# In[89]:


df_csv = spark.read.csv("C:\\Users\\dbda\\cars_data.csv", header=True, inferSchema=True)


# In[90]:


df_csv.show()


# In[ ]:


#Spark needs to understand the structure of the data, 
#including the types of each column (e.g., string, integer, float)
#By default, Spark treats all columns as strings when reading CSV files, which may not always be what you want.


# In[92]:


df_csv.printSchema()


# In[93]:


df=df_csv


# In[94]:


df.printSchema()


# In[95]:


df.select("name").show()


# In[96]:


df.select("name", "company").show()


# In[97]:


df=spark.read.csv("C:\\Users\\dbda\\synthetic_house_prices_with_categorical.csv",header=True, inferSchema=True )


# In[98]:


df.show()


# In[99]:


df.printSchema()


# In[100]:


df.filter(df["Area"] > 575).show()


# In[101]:


df.filter(df["Area"] > 459 ).select("House_Type","Area").show()


# In[102]:


df.withColumn("add numbers inside it", df["Area"]+1000).select("House_Type").show()


# In[103]:


df.withColumn("add numbers inside it", df["Area"]+1000).select("House_Type","Area").show()


# In[104]:


df.select("Area").show()


# In[105]:


df.withColumnRenamed("Area", "Years").show()


# In[ ]:





# In[106]:


df.show()


# In[107]:


df.drop("Zip_Code","Has_Garden").show()


# In[108]:


df.groupBy("House_Type").count().show()


# In[109]:


df.groupBy("House_Type").avg({"House_Type":"avg"}).show()


# In[111]:


df.groupBy("Age").agg({"Age": "avg"}).show()


# In[112]:


df.orderBy("Area").select("Area").show()


# In[113]:


df.orderBy(df["Area"].desc()).select("Area").show()


# In[114]:


df1=spark.createDataFrame([("Alice", 1),("Bob", 2),("Bob", 2)], ["Name", "ID"])
df2=spark.createDataFrame([("Alice", "HR"),("Bob", "Engineering")], ["Name", "Department"])


# In[115]:


df2


# In[116]:


df1.join(df2, "Name", "inner").show()


# In[119]:


df1 = spark.createDataFrame([("Alice", 1), ("Bob", 2)], ["Name", "ID"])
df2 = spark.createDataFrame([("Alice", "HR"), ("Bob", "Engineering")], ["Name", "Department"])

# Inner Join
df1.join(df2, "Name", "inner").show()

# Left Join
df1.join(df2, "Name", "left").show()


# In[118]:


import os
os.environ["PYSPARK_PYTHON"] = "python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3.9"


# In[120]:


import os
os.environ["PYSPARK_PYTHON"] = "python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3.9"

from pyspark.sql import SparkSession

# Start your Spark session
spark = SparkSession.builder.appName("Example").getOrCreate()

# Now run your join
df1.join(df2, "Name", "inner").show()


# In[121]:


rdd= spark.sparkContext.parallelize([1,2,3,4,5])


# In[122]:


rdd.collect()


# In[126]:


rdd2 = rdd.map(lambda x : x*2)
rdd2
#use rdd.collect()


# In[128]:


rdd.reduce(lambda a,b : a+b)


# In[129]:


#writig a data to files


# In[132]:


#df.write.csv("output_path.csv", header=True)
#df.write.parquet("output_path.parquet")
#df.write.json("output_path.json")


# In[134]:


df.show()


# In[135]:


df.createOrReplaceTempView("people")


# In[136]:


df


# In[138]:


# Incorrect (causing the error)
sqark = SparkSession.builder.appName("MyApp").getOrCreate()

# Correct
spark = SparkSession.builder.appName("MyApp").getOrCreate()


# In[139]:


sqark.sql("SELECT * FROM people Where No_of_Rooms > 2").show()


# In[143]:


query="SELECT * FROM people Where No_of_Rooms > 2"
result = spark.sql(query)
result.show()


# In[144]:


from pyspark.sql import functions as F

# Example: broadcasting a smaller DataFrame
df_small = spark.createDataFrame([("Alice", 10), ("Bob", 20)], ["Name", "Value"])
df_large = spark.createDataFrame([("Alice", "HR"), ("Bob", "Engineering")], ["Name", "Department"])

broadcast_df_small = spark.sparkContext.broadcast(df_small.collect())
# Use broadcast_df_small for joins or other operations


# In[145]:


df.persist()  # Keep the DataFrame in memory
df.show()

# To un-persist and release memory
df.unpersist()


# In[146]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# Example DataFrame
data = [(1, 1), (2, 3), (3, 5), (4, 7), (5, 9)]
columns = ["X", "Y"]
df = spark.createDataFrame(data, columns)

# Feature Engineering: Combine features into a single vector
assembler = VectorAssembler(inputCols=["X"], outputCol="features")
df = assembler.transform(df)

# Fit the model
lr = LinearRegression(featuresCol="features", labelCol="Y")
model = lr.fit(df)

# Make predictions
predictions = model.transform(df)
predictions.show()

