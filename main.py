import csv
# from utils import map_genre
import json
import os
from typing import List

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from surprise import Dataset
from surprise import KNNBasic
from surprise import Reader
from surprise import dump

from item_based_recommend import initial_content_based_rec, get_similar_items

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================DATA=========================
data = pd.read_csv("movie_info.csv")
movies_genres_df = pd.read_csv('movies_genres.csv')
ratings_with_one_hot = pd.read_csv('ratings_with_one_hot.csv')

"""
=================== Body =============================
"""


class Movie(BaseModel):
    movie_id: int
    movie_title: str
    release_date: str
    score: int


# == == == == == == == == == API == == == == == == == == == == =

# show top-10 genres
@app.get("/api/genre")
def get_genre():
    # content-based population profile
    ratings = ratings_with_one_hot['ratings']
    genres_df = ratings_with_one_hot.drop(columns=['userId', 'movie_id', 'ratings'])
    pop_genres_df = genres_df.T.dot(ratings)
    pop_genres_df = pop_genres_df / sum(pop_genres_df.values)
    pop_genres_df = pop_genres_df.sort_values(ascending=False)

    # top 10 generes
    a = pd.DataFrame(pop_genres_df[:10])
    pop_genres_list = a.index.tolist()

    return {'genre': pop_genres_list}


# top-12 content-based similar movies
@app.post("/api/movies")
def get_movies(genre: list):
    # generate content-based recommendation index list
    rec_index = initial_content_based_rec(movies_genres_df, genre, num=12)

    # query
    results = pd.merge(rec_index, data, on='movie_id', how='left')
    results = results.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url']]
    results.loc[:, 'score'] = None

    return json.loads(results.to_json(orient="records"))


@app.post("/api/recommend")
def get_recommend(movies: List[Movie]):
    # print(movies)
    iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)
    res = get_initial_items(iid, score)
    res = [int(i) for i in res]
    if len(res) > 12:
        res = res[:12]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


@app.post("/api/user_recommend")
def get_user_recommend(movies: List[Movie]):
    print("movies")

    return json.loads(results.to_json(orient="records"))


@app.get("/api/add_recommend/{item_id}")
async def add_recommend(item_id):
    res = get_similar_items(str(item_id), n=5)
    res = [int(i) for i in res]
    print(res)
    rec_movies = data.loc[data['movie_id'].isin(res)]
    print(rec_movies)
    rec_movies.loc[:, 'like'] = None
    results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


@app.get("/api/add_user_recommend/{item_id}")
async def add_recommend(item_id):
    # res = get_similar_items(str(item_id), n=5)
    # res = [int(i) for i in res]
    # print(res)
    # rec_movies = data.loc[data['movie_id'].isin(res)]
    # print(rec_movies)
    # rec_movies.loc[:, 'like'] = None
    # results = rec_movies.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url', 'like']]
    return json.loads(results.to_json(orient="records"))


def user_add(iid, score):
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    df.to_csv('new_' + 'u.data')
    with open(r'new_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        data_input = []
        s = [user, str(iid), int(score), '0']
        data_input.append(s)
        for k in data_input:
            wf.writerow(k)


def get_initial_items(iid, score, n=12):
    res = []
    user_add(iid, score)
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = KNNBasic(sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    dump.dump('./model', algo=algo, verbose=1)
    all_results = {}
    for i in range(1682):
        uid = str(944)
        iid = str(i)
        pred = algo.predict(uid, iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res
