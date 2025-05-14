import pandas as pd
import time

start_time = time.time()
# Load Parquet files using pandas
tracking = pd.read_parquet('tracking.parquet')
games = pd.read_parquet('games.parquet')
players = pd.read_parquet('players.parquet')
plays = pd.read_parquet('plays.parquet')

# Rename column to avoid conflict
players = players.rename(columns={"nflId": "player_nflId"})

# Merge dataframes
df_merged = tracking.merge(plays, on=["gameId", "playId"], how="left")
df_merged = df_merged.merge(games, on="gameId", how="left")
df_merged = df_merged.merge(players, on="displayName", how="left")

# Save merged DataFrame as a Parquet file

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")