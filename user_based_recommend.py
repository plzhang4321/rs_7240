import os

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import Reader
from surprise import dump
from surprise.model_selection import train_test_split

import csv

def user_add(iid, score):
    user = '944'
    # simulate adding a new user into the original data file
    df = pd.read_csv('./u.data')
    df.to_csv('new_' + 'u.data')
    with open(r'new_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        data_input = []
        for i in range(len(iid)):
            s = [user, str(iid[i]), int(score[i]), '0']
            data_input.append(s)
        for k in data_input:
            wf.writerow(k)

def add_new_data(user_id, movie_id):
    with open(r'new_u.data', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa, delimiter='\t')
        #user like the movie, so define like as rating score 4
        wf.writerow([str(user_id),str(movie_id),4,'0'])

#function to get initial recommend list based on user first rating
#import movie id list and their score of user first rating, num deside number of top output element
def user_init_rec(movie_id, rating, num=24):
    rec_list = []
    #add new user rating data to file new_u.data
    user_add(movie_id, rating)
    #train recommend model
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset, testset = train_test_split(data, test_size=.9)
    algo = KNNWithMeans(k=10,sim_options={'name': 'pearson', 'user_based': True})
    algo.fit(trainset)
    dump.dump('./model', algo=algo, verbose=1)
    #predict user rating for exsit movie
    all_results = {}
    for i in range(1682):
        uid = str(944)
        iid = str(i)
        pred = algo.predict(uid, iid).est
        all_results[iid] = pred
    #sort rating result for big to small, get top n for recommend
    sorted_list = sorted(all_results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for i in range(num):
        rec_list.append(sorted_list[i][0])
    return rec_list

def refine_recommend(user_id,movie_id,rec_list):
    #add new user feature to dataset
    add_new_data(user_id, movie_id)
    #predict recommend list
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset, testset = train_test_split(data, test_size=.9)
    algo = KNNWithMeans(k=20, sim_options={'name': 'pearson', 'user_based': True})
    algo.fit(trainset)
    inner_id = algo.trainset.to_inner_iid(user_id)
    #ensure recommend list include at least 5 items except user feature
    n = len(rec_list)
    neighbors = algo.get_neighbors(inner_id, k=n+5+len(rec_list))
    neighbors_movie_id = [algo.trainset.to_raw_iid(x) for x in neighbors]
    #print("test_neighbour" + str(neighbors_movie_id))
    return neighbors_movie_id