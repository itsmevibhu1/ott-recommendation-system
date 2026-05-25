import pandas as pd
import numpy as np
import random

np.random.seed(42)

# -------------------------
# USERS
# -------------------------
users = [f"U{i}" for i in range(1, 101)]

# -------------------------
# MOVIES
# -------------------------
genres = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi"]
languages = ["English", "Hindi", "Korean"]

movies = []

for i in range(1, 51):
    movies.append({
        "movie_id": f"M{i}",
        "title": f"Movie_{i}",
        "genre": random.choice(genres),
        "language": random.choice(languages),
        "popularity": round(np.random.uniform(1, 10), 2)
    })

movies_df = pd.DataFrame(movies)

# -------------------------
# USER EVENTS
# -------------------------
events = []

for _ in range(3000):
    user = random.choice(users)
    movie = random.choice(movies)

    watch_percent = np.random.randint(10, 100)
    rewatch = np.random.choice([0, 1], p=[0.8, 0.2])
    early_exit = 1 if watch_percent < 30 else 0
    preview_depth = round(np.random.uniform(0, 1), 2)

    events.append({
        "user_id": user,
        "movie_id": movie["movie_id"],
        "watch_percent": watch_percent,
        "rewatch": rewatch,
        "early_exit": early_exit,
        "preview_depth": preview_depth,
        "searched_genre": random.choice(genres)
    })

events_df = pd.DataFrame(events)

# Create label
events_df["label"] = (events_df["watch_percent"] > 70).astype(int)

# Save CSVs
movies_df.to_csv("movies.csv", index=False)
events_df.to_csv("user_events.csv", index=False)

print("Dummy data created successfully!")
print(events_df.head())
print(movies_df.head())

