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
    distances = {}
    selected_vector = tfidf_matrix[selected_movie]

    for idx, _ in df.iterrows():
        if idx != selected_movie:
            distance = euclidean(selected_vector.toarray(), tfidf_matrix[idx].toarray())
            distances[idx] = distance
    
    recommended_indices = sorted(distances, key=distances.get)[:n_recommendations]

    # Return both movieId and original_title
    recommended_movies = df.loc[recommended_indices, ["movieId"]]["movieId"]
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

    recommendations = recommend_movies_euclidean(
        selectedIndex,
        movies_df,
        n_recommendations=n_recommendations,
        tfidf_matrix=tfidf_matrix
    )
    print("Recommendations:", recommendations)

    return Response(list(recommendations))
