import pandas as pd
import fastparquet
import pyarrow
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
spark = SparkSession.builder.getOrCreate()
df_merged = spark.read.parquet('merged_tracking_data.parquet')

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

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

# Step 1: Join rusher info (assumes plays_df has nflIdRusher)


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

ball_snap_frames = df_merged.filter(col("event") == "ball_snap") \
    .groupBy("gameId", "playId") \
    .agg(max("frameId").alias("ball_snap_frame"))


before_snap_frames = ball_snap_frames.withColumn("before_snap_frame", col("ball_snap_frame") - 1)

df_before_snap = df_merged.join(before_snap_frames, on=["gameId", "playId"]) \
    .filter(col("frameId") == col("before_snap_frame"))

# Step 1: Filter OL + TE at before ball_snap on run plays
positions_all = ["T", "G", "C", "TE"]
df_all = df_before_snap.filter(
    (col("playType") == "run") &
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
df_all = df_all.withColumn("x_rel_los", col("x") - col("absoluteYardlineNumber"))

# Step 2: Compute x relative to LOS
df_all = df_all.withColumn("x_rel_los", col("x") - (col("absoluteYardlineNumber") + 0))

# Step 3: Add run direction label
df_all = df_all.withColumn(
    "run_dir",
    when(col("rushLocationType").contains("LEFT"), "left")
    .when(col("rushLocationType").contains("RIGHT"), "right")
)

# Step 4: Assign true positions
window = Window.partitionBy("gameId", "playId", "position").orderBy("y")
df_ranked = df_all.withColumn("position_index", F.row_number().over(window))

df_final = df_ranked.withColumn(
    "true_position",
    when((col("position") == "T") & (col("position_index") == 1), "LT")
    .when((col("position") == "T") & (col("position_index") == 2), "RT")
    .when((col("position") == "G") & (col("position_index") == 1), "LG")
    .when((col("position") == "G") & (col("position_index") == 2), "RG")
    .when(col("position") == "C", "C"))


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
pivot_positions = ["LT", "LG", "C", "RG", "RT"]
df_wide = (
    df_with_yrel.select("gameId", "playId", "run_dir", "true_position", "x_rel_los", "y_rel_C", "s", "dir", "o", "a", "dis")
    .filter(col("true_position").isin(pivot_positions))
    .groupBy("gameId", "playId", "run_dir")
    .pivot("true_position", pivot_positions)
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
X = df_model.drop(columns=["gameId", "playId", "run_dir"])
y = df_model["run_dir"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models_and_params = [
    ("RandomForest", RandomForestClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10]
    }),
    ("XGBoost", XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.3]
    }),
    ("LogReg", LogisticRegression(max_iter=1000), {
        "C": [0.1, 1, 10]
    }),
    ("GradientBoosting", GradientBoostingClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "learning_rate": [0.1, 0.3],
        "max_depth": [3, 5]
    }),
]

best_model = None
best_score = 0
best_name = ""
best_grid = None

for name, model, params in models_and_params:
    print(f"\nTraining {name}...")
    grid = GridSearchCV(model, params, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    score = accuracy_score(y_test, grid.predict(X_test))
    print(f"{name} accuracy: {score:.2%} | Best params: {grid.best_params_}")

    if score > best_score:
        best_model = grid.best_estimator_
        best_score = score
        best_name = name
        best_grid = grid

print(f"\n✅ Best Model: {best_name}")
print(f"Accuracy on Test Set: {best_score:.2%}")
print("Best Parameters:", best_grid.best_params_)
