import pandas as pd
import joblib

# -------------------------
# LOAD FILES
# -------------------------
model = joblib.load("ranker.pkl")

candidates = pd.read_csv(
    "candidate_recommendations.csv"
)

movies = pd.read_csv(
    "movies.csv"
)

# -------------------------
# CHOOSE USER
# -------------------------
user_id = "U1"

# -------------------------
# GET USER CANDIDATES
# -------------------------
user_candidates = candidates[
    candidates["user_id"] == user_id
].copy()

# Add movie popularity
user_candidates = user_candidates.merge(
    movies[["movie_id", "title", "popularity"]],
    on="movie_id",
    how="left"
)

# -------------------------
# MODEL FEATURES
# -------------------------
X = user_candidates[
    ["similarity_score", "popularity"]
]

# -------------------------
# PREDICT RANK SCORES
# -------------------------
user_candidates["rank_score"] = model.predict(X)

# Sort descending
recommendations = user_candidates.sort_values(
    "rank_score",
    ascending=False
)

# Top 5
top5 = recommendations.head(5)

print("\nTop Recommendations for", user_id)
print(top5[
    ["movie_id", "title", "rank_score"]
])