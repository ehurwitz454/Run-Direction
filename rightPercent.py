import pandas as pd
import fastparquet
import pyarrow
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
# Load data
df_merged = spark.read.parquet('merged_tracking_data.parquet')
df_merged = df_merged.withColumn(
    "playType",
    when(col("passResult").isNotNull(), "pass")
    .when(col("rushLocationType").isNotNull(), "run")
)


from pyspark.sql.functions import col, count, when

# Filter to only run plays that are LEFT or RIGHT (exclude middle/unknown)
df_lr_runs = df_merged.filter(
    (col("playType") == "run") &
    (col("rushLocationType").rlike("LEFT|RIGHT"))
)

# Count total LEFT and RIGHT runs
df_dir_counts = df_lr_runs.withColumn(
    "run_right", when(col("rushLocationType").contains("RIGHT"), 1).otherwise(0)
).agg(
    count("*").alias("total_lr_runs"),
    F.sum("run_right").alias("right_runs")
)

# Compute percentage of right runs
df_dir_counts = df_dir_counts.withColumn(
    "percent_right", (col("right_runs") / col("total_lr_runs")) * 100
)

df_dir_counts.show()
