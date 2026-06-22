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

#print(result_recommendtion.head(20))

#--------------------create user base score              -----(2)-----

movie_df = movie_df.drop(columns=["genres"])
sim_with_users = rating_df[rating_df["movieId"].isin(user_info["movieId"])]
group_sim_with_users = sim_with_users.groupby(by="userId")
users = sim_with_users["userId"].unique()
pearsoncorlltion_dict = dict()
test = 0
for user in users:
    y_user = group_sim_with_users.get_group(user)
    y_mean = y_user["rating"].mean()
    x_user = user_info[user_info["movieId"].isin(y_user["movieId"])].drop(columns=["title", "genres", "year"])
    x_mean = y_user["rating"].mean()

    sxy = ((x_user["rating"].values-x_mean)*(y_user["rating"].values-y_mean)).sum()
    sxx = pow(x_user["rating"].values-x_mean, 2).sum()
    syy = pow(y_user["rating"].values-y_mean, 2).sum()

    if sxx != 0 and syy != 0:
        pearsoncorlltion_dict[int(user)] = float(sxy / sqrt(sxx*syy))
    else:
        pearsoncorlltion_dict[int(user)] = 0

pearsoncorlltion_df = pd.DataFrame.from_dict(pearsoncorlltion_dict, orient="index", columns=["Similarity"])
pearsoncorlltion_df["userId"] = pearsoncorlltion_df.index
pearsoncorlltion_df = pearsoncorlltion_df.sort_values(by="Similarity", ascending=False)

best_similarity_users = pearsoncorlltion_df.merge(rating_df, on="userId")
best_similarity_users = best_similarity_users[best_similarity_users["Similarity"] > 0]
best_similarity_users["weigh"] = best_similarity_users["Similarity"]*best_similarity_users["rating"]

sum_of_best_similarity_users = best_similarity_users.groupby(by="movieId").sum()[["Similarity", "rating", "weigh"]]
sum_of_best_similarity_users["average"] = sum_of_best_similarity_users["weigh"] / sum_of_best_similarity_users["Similarity"]

result_recommendtion_user_base = sum_of_best_similarity_users.sort_values(by="average", ascending=False).drop(columns=["Similarity", "rating", "weigh"])
result_recommendtion_user_base["movieId"] = result_recommendtion_user_base.index
result_recommendtion_user_base.index = range(len(result_recommendtion_user_base))
result_recommendtion_user_base = result_recommendtion_user_base.merge(movie_df, on="movieId")

print(result_recommendtion_user_base.head(20))