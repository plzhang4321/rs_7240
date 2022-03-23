import os

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset
from surprise import KNNWithMeans
from surprise import Reader


def initial_content_based_rec(movies_genres_df, genre, num=10):
    genres_df = movies_genres_df.drop(columns='movie_id')
    genres_matrix = genres_df.to_numpy()

    # user profile one-hot
    genres_list = genres_df.columns.tolist()
    user_profile = pd.DataFrame([len(genres_list) * [0]], columns=genres_list)
    user_profile[genre] = 1

    # similarity
    u_v_matrix = user_profile.values
    similarity = cosine_similarity(u_v_matrix, genres_matrix)
    recommendation_table_df = movies_genres_df[['movie_id']].copy(deep=True)
    recommendation_table_df['similarity'] = similarity[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)
    rec_index = rec_result['movie_id'][:num]
    return rec_index


def get_similar_items(iid, n=12):
    file_path = os.path.expanduser('new_u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset = data.build_full_trainset()
    algo = KNNWithMeans(k=20, sim_options={'name': 'pearson', 'user_based': False})
    algo.fit(trainset)
    inner_id = algo.trainset.to_inner_iid(iid)
    print("test inner_id" + str(inner_id))
    neighbors = algo.get_neighbors(inner_id, k=n)
    neighbors_iid = [algo.trainset.to_raw_iid(x) for x in neighbors]
    print("test_neighbour" + str(neighbors_iid))
    return neighbors_iid
