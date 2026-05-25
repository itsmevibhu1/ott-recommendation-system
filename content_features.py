import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# -------------------------
# LOAD MOVIE DATA
# -------------------------
movies = pd.read_csv("movies.csv")

# -------------------------
# SELECT FEATURES
# -------------------------
feature_cols = ["genre", "language"]

# -------------------------
# ONE-HOT ENCODE
# Converts text → numbers
# -------------------------
encoder = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(sparse_output=False),
            feature_cols
        )
    ],
    remainder="passthrough"
)

# Fit and transform
movie_matrix = encoder.fit_transform(
    movies[["genre", "language", "popularity"]]
)

# Get feature names
encoded_names = encoder.named_transformers_[
    "cat"
].get_feature_names_out(feature_cols)

# Build dataframe
movie_features = pd.DataFrame(
    movie_matrix,
    columns=list(encoded_names) + ["popularity"]
)

# Add movie_id
movie_features.insert(
    0,
    "movie_id",
    movies["movie_id"]
)

# -------------------------
# SAVE
# -------------------------
movie_features.to_csv(
    "movie_features.csv",
    index=False
)

print("Movie features created successfully!")
print(movie_features.head())