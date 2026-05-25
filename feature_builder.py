import pandas as pd

# -------------------------
# LOAD DATA
# -------------------------
events = pd.read_csv("user_events.csv")
movies = pd.read_csv("movies.csv")

# Merge movie genre into events
df = events.merge(
    movies[["movie_id", "genre"]],
    on="movie_id",
    how="left"
)

# -------------------------
# BASIC USER FEATURES
# -------------------------
user_features = df.groupby("user_id").agg(
    avg_completion_rate=("watch_percent", "mean"),
    rewatch_rate=("rewatch", "mean"),
    early_exit_rate=("early_exit", "mean"),
    avg_preview_depth=("preview_depth", "mean"),
    total_watches=("movie_id", "count")
).reset_index()

# -------------------------
# GENRE AFFINITY FEATURES
# -------------------------
# Average watch % per genre per user
genre_affinity = df.pivot_table(
    index="user_id",
    columns="genre",
    values="watch_percent",
    aggfunc="mean",
    fill_value=0
)

# Rename columns
genre_affinity.columns = [
    f"{col}_affinity" for col in genre_affinity.columns
]

genre_affinity = genre_affinity.reset_index()

# -------------------------
# COMBINE FEATURES
# -------------------------
final_features = user_features.merge(
    genre_affinity,
    on="user_id",
    how="left"
)

# -------------------------
# SAVE
# -------------------------
final_features.to_csv(
    "user_features.csv",
    index=False
)

print("User features created successfully!")
print(final_features.head())