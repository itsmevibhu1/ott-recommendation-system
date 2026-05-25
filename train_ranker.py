import pandas as pd
import joblib
from lightgbm import LGBMRanker

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("ranking_data.csv")

# -------------------------
# FEATURES
# -------------------------
X = df[
    ["similarity_score", "popularity"]
]

y = df["label"]

# Group sizes:
# how many rows per user
group = df.groupby(
    "user_id"
).size().to_list()

# -------------------------
# TRAIN MODEL
# -------------------------
model = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=100,
    learning_rate=0.1
)

model.fit(
    X,
    y,
    group=group
)

# -------------------------
# SAVE MODEL
# -------------------------
joblib.dump(
    model,
    "ranker.pkl"
)

print("Ranking model trained!")