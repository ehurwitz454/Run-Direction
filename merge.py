def spark_merge(df_tracking, df_plays, df_players, df_games):
    pass

import fastparquet
import pyarrow
import pandas as pd
from pandas import read_parquet
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time
start_time = time.time()
# Set driver memory before creating the session
spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# Load Parquet files
tracking = spark.read.parquet('tracking.parquet')
games = spark.read.parquet('games.parquet')
players = spark.read.parquet('players.parquet')
plays = spark.read.parquet('plays.parquet')

# Rename column to avoid conflict
players = players.withColumnRenamed("nflId", "player_nflId")

# Merge dataframes
df_merged = tracking.join(plays, on=["gameId", "playId"], how="left")
df_merged = df_merged.join(games, on="gameId", how="left")
df_merged = df_merged.join(players, on="displayName", how="left")

# Optional: repartition to avoid memory issues
df_merged = df_merged.repartition(500)

# Save as Parquet
df_merged.write.mode("overwrite").parquet("merged_tracking_data.parquet")
end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")