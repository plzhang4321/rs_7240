from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import csv
from sklearn.cluster import estimate_bandwidth
from surprise import Reader
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
# from utils import map_genre
import json
from surprise import dump
from surprise import KNNBasic
from surprise import Dataset

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
movies_genres_df=pd.read_csv('movies_genres.csv')
ratings_with_one_hot=pd.read_csv('ratings_with_one_hot.csv')

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
    #content-based population profile
    ratings=ratings_with_one_hot['ratings']
    genres_df=ratings_with_one_hot.drop(columns=['userId','movie_id','ratings'])
    pop_genres_df=genres_df.T.dot(ratings)
    pop_genres_df=pop_genres_df/sum(pop_genres_df.values)
    pop_genres_df=pop_genres_df.sort_values(ascending=False)

    #top 10 generes
    a=pd.DataFrame(pop_genres_df[:10])
    pop_genres_list=a.index.tolist()

    return {'genre':pop_genres_list}


# top-12 content-based similar movies
@app.post("/api/movies")
def get_movies(genre: list):
    #generate content-based recommendation index list
    rec_index=initial_content_based_rec(movies_genres_df,genre,num=12)

    #query
    results=pd.merge(rec_index,data,on='movie_id', how='left')
    results = results.loc[:, ['movie_id', 'movie_title', 'release_date', 'poster_url']]
    results.loc[:, 'score'] = None

    return json.loads(results.to_json(orient="records"))


@app.post("/api/recommend")
def get_recommend(movies: List[Movie]):
    # print(movies)
    iid = str(sorted(movies, key=lambda i: i.score, reverse=True)[0].movie_id)
    score = int(sorted(movies, key=lambda i: i.score, reverse=True)[0].score)
    res = get_initial_items(iid,score)
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

#for dialog 0 & 1
def initial_content_based_rec(movies_genres_df,genre,num=10):
    genres_df=movies_genres_df.drop(columns='movie_id')
    genres_matrix=genres_df.to_numpy()

   #user profile one-hot
    genres_list=genres_df.columns.tolist()
    user_profile=pd.DataFrame([len(genres_list)*[0]],columns=genres_list)
    user_profile[genre]=1

    #similarity
    u_v_matrix=user_profile.values
    similarity=cosine_similarity(u_v_matrix,genres_matrix)
    recommendation_table_df = movies_genres_df[['movie_id']].copy(deep=True)
    recommendation_table_df['similarity'] = similarity[0]
    rec_result= recommendation_table_df.sort_values(by=['similarity'], ascending=False)
    rec_index=rec_result['movie_id'][:num]
    return rec_index

def user_add(iid, score):
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    df.to_csv('new_' + 'u.data')
    with open(r'new_u.data',mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa,delimiter='\t')
        data_input = []
        s = [user,str(iid),int(score),'0']
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
    dump.dump('./model',algo=algo,verbose=1)
    all_results = {}
    for i in range(1682):
        uid = str(944)
        iid = str(i)
        pred = algo.predict(uid,iid).est
        all_results[iid] = pred
    sorted_list = sorted(all_results.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    for i in range(n):
        print(sorted_list[i])
        res.append(sorted_list[i][0])
    return res

def get_similar_items(iid, n=12):
    algo = dump.load('./model')[1]
    inner_id = algo.trainset.to_inner_iid(iid)
    print(inner_id)
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print(neighbors_iid)
    return neighbors_iid
