import pandas as pd
import fastparquet
import pyarrow
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
# Load data
df_merged = spark.read.parquet('merged_tracking_data.parquet')
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import when, col

# Create playType: "pass", "run", or null
df_merged = df_merged.withColumn(
    "playType",
    when(col("passResult").isNotNull(), "pass")
    .when(col("rushLocationType").isNotNull(), "run")
)


from pyspark.sql.functions import col, when
from pyspark.sql import Window
import pyspark.sql.functions as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Step 1: Filter OL + TE at ball_snap on run plays
positions_all = ["T", "G", "C", "TE"]
df_all = df_merged.filter(
    (col("playType") == "run") &
    (col("event") == "line_set") &
    (col("position").isin(positions_all)) &
    (col("rushLocationType").isNotNull()) &
    (col("absoluteYardlineNumber").isNotNull()) &
    (col("x").isNotNull()) &
    (col("y").isNotNull()) &
    (col("s").isNotNull()) &
    (col("dir").isNotNull()) &
    (col("o").isNotNull()) &
    (col("a").isNotNull()) &
    (col("dis").isNotNull())
)

# Step 2: Compute x relative to LOS
df_all = df_all.withColumn("x_rel_los", col("x") - (col("absoluteYardlineNumber") + 0))

# Step 3: Add run direction label
df_all = df_all.withColumn(
    "run_dir",
    when(col("rushLocationType").contains("LEFT"), "left")
    .when(col("rushLocationType").contains("RIGHT"), "right")
)
# Count number of TEs per play
df_te_counts = df_all.filter(col("position") == "TE") \
    .groupBy("gameId", "playId") \
    .agg(F.count("*").alias("num_TEs"))

# Join count back to main filtered DataFrame
df_all_with_te = df_all.join(df_te_counts, on=["gameId", "playId"], how="left").fillna(0)

# Step 4: Assign true positions
window = Window.partitionBy("gameId", "playId", "position").orderBy("y")
df_ranked = df_all_with_te.withColumn("position_index", F.row_number().over(window))

df_final = df_ranked.withColumn(
    "true_position",
    when((col("position") == "T") & (col("position_index") == 1), "LT")
    .when((col("position") == "T") & (col("position_index") == 2), "RT")
    .when((col("position") == "G") & (col("position_index") == 1), "LG")
    .when((col("position") == "G") & (col("position_index") == 2), "RG")
    .when(col("position") == "C", "C")
    .when(col("position") == "TE", "TE"))


# Step 5: Keep resolved positions
df_filtered = df_final.filter(col("true_position").isNotNull())

# Step 6: Get center y per play
df_c_y = df_filtered.filter(col("true_position") == "C") \
    .select("gameId", "playId", col("y").alias("center_y"))

# Step 7: Join center y and compute relative y
df_with_c_y = df_filtered.join(df_c_y, on=["gameId", "playId"])
df_with_yrel = df_with_c_y.withColumn(
    "y_rel_C",
    when(col("true_position") == "C", col("y"))
    .otherwise(col("y") - col("center_y"))
)

# Step 8: Pivot features
# After `df_with_yrel` has been computed...
df_with_yrel = df_with_yrel.withColumn("num_TEs", col("num_TEs").cast("int"))

df_wide = (
    df_with_yrel.select("gameId", "playId", "run_dir", "offenseFormation", "possessionTeam", "num_TEs", "true_position",
                        "x_rel_los", "y_rel_C", "s", "dir", "o", "a", "dis")
    .filter(col("true_position").isin(["LT", "LG", "C", "RG", "RT", "TE"]))
    .groupBy("gameId", "playId", "run_dir", "possessionTeam", "offenseFormation", "num_TEs")
    .pivot("true_position", ["LT", "LG", "C", "RG", "RT", "TE"])
    .agg(
        F.first("y_rel_C").alias("y"),
        F.first("dir").alias("dir"),
        F.first("o").alias("o")
    )
    .dropna()
)

# Step 9: Train model
df_model = df_wide.toPandas()
for pos in ["LT", "LG", "C", "RG", "RT", "TE"]:  # Only include TE if it exists
    dir_col = f"{pos}_dir"
    o_col = f"{pos}_o"
    if dir_col in df_model.columns and o_col in df_model.columns:
        df_model[f"{pos}_dir_o"] = df_model[dir_col] * df_model[o_col]
X = df_model.drop(columns=["gameId", "playId", "run_dir", "possessionTeam", "offenseFormation", "num_TEs"])
y = df_model["run_dir"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

results = []

for n_te in [0, 1, 2]:
    df_te = df_model[df_model["num_TEs"] == n_te]

    if len(df_te) < 100:
        continue  # skip if not enough data

    X = df_te.drop(columns=["gameId", "playId", "run_dir", "possessionTeam", "offenseFormation", "num_TEs"])
    y = le.transform(df_te["run_dir"])

    # Optional: engineer new features (e.g., TE_dir * TE_o if TE features exist)
    if "TE_dir" in X.columns and "TE_o" in X.columns:
        X["TE_dir_o"] = X["TE_dir"] * X["TE_o"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(
        n_estimators=190,
        max_depth=11,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="log2",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    results.append((n_te, acc))
print("\n📊 Accuracy by Number of TEs:")
for n_te, acc in results:
    print(f"{n_te} TEs: Accuracy = {acc:.4f}")

team_accuracies = []

for team in df_model["possessionTeam"].unique():
    df_team = df_model[df_model["possessionTeam"] == team]

    if len(df_team) < 100:
        continue  # Skip teams with too few samples for meaningful split

    X_team = df_team.drop(columns=["gameId", "playId", "run_dir", "possessionTeam", "offenseFormation"])
    y_team = le.transform(df_team["run_dir"])  # reuse fitted label encoder

    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X_team, y_team, test_size=0.25, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=190,
        max_depth=11,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="log2",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_t, y_train_t)
    accuracy = clf.score(X_test_t, y_test_t)

    team_accuracies.append((team, accuracy))

# Convert to DataFrame and rank
df_acc = pd.DataFrame(team_accuracies, columns=["Team", "Accuracy"])
df_acc = df_acc.sort_values("Accuracy", ascending=False).reset_index(drop=True)

print("\n📊 Accuracy by Offensive Team:")
print(df_acc.to_string(index=False))

# Convert Spark to pandas (if needed)
df_rushing = df_merged.filter(
    (col("playType") == "run") &
    (col("rushLocationType").isNotNull()) &
    (col("yardsGained").isNotNull())
).select("possessionTeam", "rushLocationType", "yardsGained")

# Aggregate: average yards gained by team + direction
df_rushing_avg = (
    df_rushing.withColumn(
        "run_dir", when(col("rushLocationType").contains("LEFT"), "left")
                  .when(col("rushLocationType").contains("RIGHT"), "right")
    )
    .groupBy("possessionTeam")
    .agg(F.avg("yardsGained").alias("avg_yards"))
    .toPandas()
)
# df_acc: contains Team, Accuracy
df_rushing_avg.rename(columns={"possessionTeam": "Team"}, inplace=True)
df_combined = pd.merge(df_rushing_avg, df_acc, on="Team")
import statsmodels.api as sm

# Step 1: Add constant (intercept)
X_sm = sm.add_constant(df_combined["Accuracy"])  # or df_left["Accuracy"] if you're using left only
y_sm = df_combined["avg_yards"]

# Step 2: Fit OLS model
model = sm.OLS(y_sm, X_sm).fit()

# Step 3: Print summary
print(model.summary())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(df_combined["Accuracy"], df_combined["avg_yards"], label="Teams")
plt.plot(df_combined["Accuracy"], model.predict(X_sm), color="red", label="Regression Line")
plt.xlabel("Model Accuracy")
plt.ylabel("Avg Rushing Yards per Play")
plt.title("Model Accuracy vs Avg Rushing Yards")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


formation_accuracies = []

for formation in df_model["offenseFormation"].unique():
    df_form = df_model[df_model["offenseFormation"] == formation]

    if len(df_form) < 100:
        continue  # skip formations with limited data

    X_form = df_form.drop(columns=["gameId", "playId", "run_dir", "possessionTeam", "offenseFormation"])
    y_form = le.transform(df_form["run_dir"])

    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_form, y_form, test_size=0.25, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=190,
        max_depth=11,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="log2",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_f, y_train_f)
    accuracy = clf.score(X_test_f, y_test_f)

    formation_accuracies.append((formation, accuracy))

df_form_acc = pd.DataFrame(formation_accuracies, columns=["Formation", "Accuracy"])
df_form_acc = df_form_acc.sort_values("Accuracy", ascending=False).reset_index(drop=True)

print("\n📊 Accuracy by Offensive Formation:")
print(df_form_acc.to_string(index=False))

# Step 1: Filter to run plays with formation and rush direction left/right
df_rushing_lr = df_merged.filter(
    (col("playType") == "run") &
    (col("rushLocationType").isNotNull()) &
    (col("yardsGained").isNotNull()) &
    (col("offenseFormation").isNotNull()) &
    (
        col("rushLocationType").contains("LEFT") |
        col("rushLocationType").contains("RIGHT")
    )
)

# Step 2: Group by formation and compute average yards gained
df_yards_avg = df_rushing_lr.groupBy("offenseFormation") \
    .agg(avg("yardsGained").alias("avg_yards_lr")) \
    .withColumnRenamed("offenseFormation", "Formation") \
    .orderBy("avg_yards_lr", ascending=False)

# Convert to pandas
df_yards_avg_pd = df_yards_avg.toPandas()
df_combined = pd.merge(df_form_acc, df_yards_avg_pd, on="Formation")

import statsmodels.api as sm
import matplotlib.pyplot as plt

X = sm.add_constant(df_combined["Accuracy"])
y = df_combined["avg_yards_lr"]
model = sm.OLS(y, X).fit()

print(model.summary())

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(df_combined["Accuracy"], df_combined["avg_yards_lr"], label="Formations")
plt.plot(df_combined["Accuracy"], model.predict(X), color="red", label="Regression Line")
plt.xlabel("Model Accuracy")
plt.ylabel("Avg Rushing Yards (Left or Right)")
plt.title("Model Accuracy vs Avg Rushing Yards (Left/Right)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from pyspark.sql.window import Window
from pyspark.sql.functions import col, when, avg, first, abs as ps_abs, row_number

# Step 1: Filter OL + TE at ball_snap on left/right runs
positions = ["T", "G", "C", "TE"]
df_line = df_merged.filter(
    (col("playType") == "run") &
    (col("event") == "ball_snap") &
    (col("position").isin(positions)) &
    (col("rushLocationType").rlike("LEFT|RIGHT")) &
    col("x").isNotNull() & col("y").isNotNull() &
    col("dir").isNotNull() & col("o").isNotNull() &
    col("dis").isNotNull() & col("a").isNotNull() &
    col("absoluteYardlineNumber").isNotNull()
)

# Step 2: Add run direction
df_line = df_line.withColumn(
    "run_dir",
    when(col("rushLocationType").contains("LEFT"), "left")
    .when(col("rushLocationType").contains("RIGHT"), "right")
)

# Step 3: Compute x_rel_los
df_line = df_line.withColumn("x_rel_los", col("x") - col("absoluteYardlineNumber"))

# Step 4: Identify true OL position (LT/LG/C/RG/RT)
window = Window.partitionBy("gameId", "playId", "position").orderBy("y")
df_ranked = df_line.withColumn("position_index", row_number().over(window))

df_true = df_ranked.withColumn(
    "true_position",
    when((col("position") == "T") & (col("position_index") == 1), "LT")
    .when((col("position") == "T") & (col("position_index") == 2), "RT")
    .when((col("position") == "G") & (col("position_index") == 1), "LG")
    .when((col("position") == "G") & (col("position_index") == 2), "RG")
    .when(col("position") == "C", "C")
    .when(col("position") == "TE", "TE")
).filter(col("true_position").isNotNull())

# Step 5: Compute y_rel_C
df_c_y = df_true.filter(col("true_position") == "C") \
    .select("gameId", "playId", col("y").alias("center_y"))

df_with_yrel = df_true.join(df_c_y, on=["gameId", "playId"], how="left") \
    .withColumn("y_rel_C", col("y") - col("center_y"))

# Step 6: Average features by player-position-dir
df_avg = df_with_yrel.groupBy("nflId", "displayName", "true_position", "run_dir").agg(
    avg("dir").alias("avg_dir"),
    avg("o").alias("avg_o"),
    avg("y_rel_C").alias("avg_yrel")
)

# Step 7: Pivot left/right
df_pivot = df_avg.groupBy("nflId", "displayName", "true_position").pivot("run_dir", ["left", "right"]).agg(
    first("avg_dir").alias("dir"),
    first("avg_o").alias("o"),
    first("avg_yrel").alias("yrel")
)

# Step 8: Compute differences
df_diff = df_pivot.withColumn("dir_diff", ps_abs(col("left_dir") - col("right_dir"))) \
    .withColumn("o_diff", ps_abs(col("left_o") - col("right_o"))) \
    .withColumn("yrel_diff", ps_abs(col("left_yrel") - col("right_yrel"))) \
    .withColumn("total_diff", col("dir_diff") + col("o_diff") + col("yrel_diff"))

# Step 9: Convert to pandas and display
df_leaderboard = df_diff.orderBy(col("total_diff").desc()).toPandas()
print(df_leaderboard[["displayName", "true_position", "dir_diff", "o_diff", "yrel_diff", "total_diff"]].head(20))
# Feature-specific leaderboards
feature_cols = ["dir_diff", "o_diff", "yrel_diff"]
feature_leaderboards = {}

for feature in feature_cols:
    df_feat = df_leaderboard.sort_values(feature, ascending=False).reset_index(drop=True)
    feature_leaderboards[feature] = df_feat
    print(f"\n📋 Top Players by {feature}:")
    print(df_feat[["displayName", "true_position", feature]].head(10))
