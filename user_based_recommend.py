import os

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import Reader
from surprise import dump
from surprise.model_selection import train_test_split
import random

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
    trainset, testset = train_test_split(data, test_size=.9,random_state=0)
    algo = KNNWithMeans(k=20,sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)
    dump.dump('./model', algo=algo, verbose=1)
    #predict user rating for exsit movie
    all_results = {}
    for i in range(1682):
        pred = algo.predict('944',str(i) ).est
        all_results[str(i)] = pred+random.uniform(0.0001,0.05)
    #sort rating result for big to small, get top n for recommend
    sorted_list = sorted(all_results.items(), key=lambda kv: kv[1], reverse=True)
    for i in range(num):
        rec_list.append(sorted_list[i][0])
    return rec_list

def refine_recommend(user_id,movie_id,n):
    #add new user feature to dataset
    add_new_data(user_id, movie_id)
    #predict recommend list
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset, testset = train_test_split(data, test_size=.9,random_state=0)
    algo = KNNWithMeans(k=20, sim_options={'name': 'pearson', 'user_based': True})
    algo.fit(trainset)
    #ensure recommend list include at least 5 items except user feature
    all_results = {}
    for i in range(1682):
        pred = algo.predict('944',str(i) ).est
        all_results[str(i)] = pred+random.uniform(0.00001,0.005)
    #sort rating result for big to small, get top n for recommend
    sorted_list = sorted(all_results.items(), key=lambda kv: kv[1], reverse=True)
    rec_list = []
    for i in range(n):
        rec_list.append(sorted_list[i][0])
    return rec_list