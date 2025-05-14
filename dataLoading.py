import pandas as pd
import fastparquet
import pyarrow

def load_and_merge_data():
    # Load CSV files from the current directory
    games = pd.read_csv('games.csv')
    player_play = pd.read_csv('player_play.csv')
    players = pd.read_csv('players.csv')
    plays = pd.read_csv('plays.csv')

    # Load and combine all tracking week files
    tracking_dfs = []
    for i in range(1, 10):
        tracking_dfs.append(pd.read_csv(f'tracking_week_{i}.csv'))
    tracking = pd.concat(tracking_dfs, ignore_index=True)

    return games, player_play, players, plays, tracking

games, player_play, players, plays, tracking = load_and_merge_data()
games.to_parquet('games.parquet', index=False, compression='snappy')
player_play.to_parquet('player_play.parquet', index=False, compression='snappy')
plays.to_parquet('plays.parquet', index=False, compression='snappy')
players.to_parquet('players.parquet', index=False, compression='snappy')
tracking.to_parquet('tracking.parquet', index=False, compression='snappy')
