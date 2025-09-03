from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view

from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from pathlib import Path
from django.conf import settings

from recommendations.models import Movie


# Create your views here.
def recommend_movies_euclidean(selected_movie, df, n_recommendations=3, tfidf_matrix=None):
    print(f"Selected movie index: {selected_movie}")
    print(f"DataFrame shape: {df.shape}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape if tfidf_matrix is not None else 'None'}")
    
    distances = {}
    selected_vector = tfidf_matrix[selected_movie]
    print(f"Selected vector shape: {selected_vector.shape}")

    for idx in range(len(df)):
        print(idx, selected_movie)
        if idx != selected_movie:
            distance = euclidean(selected_vector.toarray(), tfidf_matrix[idx].toarray())
            print(distance)
            distances[idx] = distance

    recommended_indices = sorted(distances, key=distances.get)[:n_recommendations]
    print(f"Recommended indices: {recommended_indices}")

    # Return movieId column from numeric indices
    recommended_movies = df.iloc[recommended_indices]["movieId"]
    print(f"Recommended movies: {recommended_movies.tolist()}")
    return recommended_movies

@api_view(['GET'])
def index(request, index=0, n_recommendations=3):
    # Load movies from DB into DataFrame
    queryset = Movie.objects.all().values()
    movies_df = pd.DataFrame(list(queryset))

    print('Movies Count: {}'.format(movies_df.shape[0]))

    # Compute row position for the selected movie
    selectedIndex = movies_df.index[movies_df['movieId'] == index][0]

    print("Movie:{}".format(movies_df['original_title'].iloc[selectedIndex]))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies_df['combined_features'])

    print("TF-IDF Matrix Shape: {}".format(tfidf_matrix.shape))

    recommendations = recommend_movies_euclidean(
        selectedIndex,
        movies_df,
        n_recommendations=n_recommendations,
        tfidf_matrix=tfidf_matrix
    )
    print("Recommendations:", recommendations)

    return Response(list(recommendations))
