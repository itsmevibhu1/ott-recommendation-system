import pandas as pd

# -------------------------
# LOAD DATA
# -------------------------
candidates = pd.read_csv(
    "candidate_recommendations.csv"
)

movies = pd.read_csv(
    "movies.csv"
)

events = pd.read_csv(
    "user_events.csv"
)

# -------------------------
# GET LABELS
# label = whether user liked movie
# -------------------------
labels = events[
    ["user_id", "movie_id", "label"]
].drop_duplicates()

# -------------------------
# MERGE EVERYTHING
# -------------------------
df = candidates.merge(
    movies[["movie_id", "popularity"]],
    on="movie_id",
    how="left"
)

df = df.merge(
    labels,
    on=["user_id", "movie_id"],
    how="left"
)

# If user never watched that movie,
# assume label = 0
df["label"] = df["label"].fillna(0)

# -------------------------
# SAVE
# -------------------------
df.to_csv(
    "ranking_data.csv",
    index=False
)

print("Ranking dataset ready!")
print(df.head())