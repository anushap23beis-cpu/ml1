
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

movies["genres"] = movies["genres"].fillna("")
movies["title_clean"] = (
    movies["title"].astype(str).str.lower()
    .str.replace(r"\s*\(\d{4}\)", "", regex=True)
    .str.strip()
)

# ---------------- Content-Based Filtering ----------------
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies["title_clean"]).drop_duplicates()

def content_recommendations(title, num=5):
    if not isinstance(title, str):
        return []
    q = title.lower().strip()
    q = __import__("re").sub(r"\s*\(\d{4}\)", "", q).strip()

    if q not in indices:
        # Partial-title fallback
        matches = movies[movies["title_clean"].str.contains(q, regex=False, na=False)]
        if matches.empty:
            return []
        idx = matches.index[0]
    else:
        idx = indices[q]

    scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    selected = [i for i, _ in scores[1:num+1]]
    return movies.iloc[selected]["title"].tolist()

# ---------------- Collaborative Filtering ----------------
user_movie_matrix = ratings.pivot_table(
    index="userId", columns="movieId", values="rating", fill_value=0
)

user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def collaborative_recommendations(user_id, num=5):
    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return []

    if user_id not in user_similarity_df.index:
        return []

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:6]
    similar_movies = user_movie_matrix.loc[similar_users.index]
    candidate_scores = similar_movies.mean().sort_values(ascending=False)

    rated = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    ids = []
    for movie_id in candidate_scores.index:
        if movie_id not in rated:
            ids.append(movie_id)
        if len(ids) >= num:
            break

    return movies[movies["movieId"].isin(ids)]["title"].tolist()[:num]

# ---------------- Hybrid Recommendation ----------------
def hybrid_recommendation(title, user_id, num=5):
    content = content_recommendations(title, num)
    collab = collaborative_recommendations(user_id, num)

    result = []
    for item in content + collab:
        if item not in result:
            result.append(item)
        if len(result) >= num:
            break
    return result
