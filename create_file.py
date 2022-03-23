import pandas as pd


# items' one-hot vectors
def item_representation_based_movie_genres(movies_df, genres):
    # First let's make a copy of the movies_df
    movies_genres_df = movies_df.copy(deep=True)
    for index, row in movies_df.iterrows():
        for genre in genres:
            if row[genre] == '1':
                movies_genres_df.at[index, genre] = 1
            else:
                movies_genres_df.at[index, genre] = 0

    # obtain a numpy matrix, which will be used for computing similarities later 
    movies_genre_matrix = movies_genres_df[genres].to_numpy()

    return movies_genres_df, movies_genre_matrix


other = ["movie_id", "movie_title", "release_date", "IMDb URL", "unknown"]
genres = ["Action", "Adventure", "Animation",
          "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
          "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]
name = other + genres

data = []
with open("u.item", 'r') as f:
    lines = f.readlines()
    for line in lines:
        info = line.strip().split("||")
        l = info[0].split("|") + info[1].split("|")
        data.append(l)
# print(data)

right = pd.read_csv("movie_poster.csv")
# print(right)
ori_frame = pd.DataFrame(data, columns=name)
# print(ori_frame.head())
ori_frame['movie_id'] = ori_frame['movie_id'].astype(int)
rs = pd.merge(ori_frame, right, on='movie_id', how='left')
# print(rs.shape)
rs.to_csv("movie_info.csv", index=None)

# one-hot genres
movies_df = ori_frame.drop(columns=['movie_title', 'release_date', 'IMDb URL', "unknown"])
movies_genres_df, movies_genre_matrix = item_representation_based_movie_genres(movies_df, genres)
for genre in genres:
    movies_genres_df[genre] = movies_genres_df[genre].astype(int)

# load u.data
file_path = 'u.data'
data_df = pd.read_csv(file_path, names=['userId', 'movie_id', 'ratings', 'timestamp'], sep='\t')
rating_df = data_df.drop(columns='timestamp')

# ratings with one-hot genres
ratings_with_one_hot = pd.merge(rating_df, movies_genres_df, on='movie_id', how='left')

# save
movies_genres_df.to_csv('movies_genres.csv', index=None)
ratings_with_one_hot.to_csv('ratings_with_one_hot.csv', index=None)
