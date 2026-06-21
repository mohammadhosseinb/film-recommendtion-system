import pandas as pd
import numpy as np
from numpy import sum, pow, sqrt
from sklearn.preprocessing import MultiLabelBinarizer

movie_df = pd.read_csv("recommedtion/recommendtion_prg/movies.csv")
rating_df = pd.read_csv("recommedtion/recommendtion_prg/ratings.csv")

user_input = [
    {"title": "Akira", "rating": 4},
    {"title": "Inception", "rating": 4.5},
    {"title": "Interstellar", "rating": 5.0},
    {"title": "John Wick", "rating": 4.0},
    {"title": "Toy Story", "rating": 3.5}
]
user_input = pd.DataFrame(user_input)

#--------------------data cleaning
movie_df["year"] = movie_df.title.str.extract(r".*\((\d+)")
movie_df["title"] = movie_df.title.str.extract(r"(.*)\s\(\d+").fillna(movie_df["title"])
movie_df["title"] = movie_df["title"].apply(lambda x: x.strip())
rating_df = rating_df.drop(columns=['timestamp'])
#--------------------create content base score              -----(1)-----
movie_df["genres"] = movie_df["genres"].str.split('|')
mlb = MultiLabelBinarizer()
mlb.fit(movie_df["genres"])
genres_matrix = pd.DataFrame(mlb.transform(movie_df["genres"]), columns=mlb.classes_, index=movie_df.movieId)

user_info = movie_df[movie_df["title"].isin(user_input["title"])].merge(user_input, on="title")
user_genres_matrix = pd.DataFrame(mlb.transform(user_info["genres"]), columns=mlb.classes_)
user_profile = user_genres_matrix.transpose().dot(user_input["rating"])

formula_recommendtion = (user_profile*genres_matrix).sum(axis=1) / user_profile.sum()
result_recommendtion = pd.DataFrame(formula_recommendtion.sort_values(ascending=False))
result_recommendtion = result_recommendtion.merge(movie_df.drop(columns=["genres"]), on="movieId")

print(result_recommendtion.head(20))