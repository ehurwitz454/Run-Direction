from data.dataLoading import load_data
from processing.merge import spark_merge
from processing.lineSet import extract_line_set_features
from analysis.beforeSnap import engineer_before_snap_features
from analysis.snapTuneOptuna import run_optuna
from analysis.Analysis import summarize_results

def main():
    print("Loading data...")
    df_tracking, df_plays, df_players, df_games = load_data()

    print("Merging datasets...")
    df_merged = spark_merge(df_tracking, df_plays, df_players, df_games)

    print("Extracting line set features...")
    df_line = extract_line_set_features(df_merged)

    print("Engineering before snap features...")
    df_snap = engineer_before_snap_features(df_line)

    print("Running model tuning...")
    best_model, results = run_optuna(df_snap)

    print("Summarizing results...")
    summarize_results(best_model, results)

if __name__ == "__main__":
    main()
