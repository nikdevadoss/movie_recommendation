import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# weighted rating 
def weighted_rating(x, m="", C=""):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

def simple_recommender():
    data = pd.read_csv('./movie_data/movies_metadata.csv', low_memory=False)

    C = data['vote_average'].mean()
    m = data['vote_count'].quantile(0.90)
    filtered_movies = data.copy().loc[data['vote_count'] >= m]

    #add weighted score to dataframe
    filtered_movies['score'] = filtered_movies.apply(weighted_rating, args= (m, C), axis=1)

    # Sort and print the top 10 recommended movies
    filtered_movies = filtered_movies.sort_values('score', ascending=False)
    return filtered_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

def content_based_recommender(movie_title):
    data = pd.read_csv('./movie_data/movies_metadata.csv', low_memory=False)
    tfidf = TfidfVectorizer(stop_words='english')

    #clean text data
    data['overview'] = data['overview'].fillna('')

    #construct tfidf matrix
    tfidf_matrix = tfidf.fit_transform(data['overview'])

    # create cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    #map movies to indices
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()

    index = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return data['title'].iloc[movie_indices]


if __name__ == "__main__":
    #simple_top_10 = simple_recommender()
    content_top_10 = content_based_recommender("The Dark Knight Rises")