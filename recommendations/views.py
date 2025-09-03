from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from recommendations.models import Movie

def recommend_movies_euclidean(selected_idx, df, n_recommendations=3, tfidf_matrix=None):
    """
    Recommend movies based on Euclidean distance of TF-IDF features.
    """
    distances = {}
    # Select the row properly as a 2D slice
    selected_vector = tfidf_matrix[selected_idx, :]

    for idx in range(tfidf_matrix.shape[0]):
        if idx != selected_idx:
            distance = euclidean(selected_vector.toarray(), tfidf_matrix[idx, :].toarray())
            print(f"Distance: {distance}")
            distances[idx] = distance

    # Get top N recommendations
    recommended_indices = sorted(distances, key=distances.get)[:n_recommendations]
    recommended_movies = df.iloc[recommended_indices][["movieId"]]
    return recommended_movies

@api_view(['GET'])
def index(request, index=0, n_recommendations=3):
    """
    API endpoint to get movie recommendations based on a movieId.
    """
    # Load all movies from DB
    queryset = Movie.objects.all().values()
    movies_df = pd.DataFrame(list(queryset))

    if movies_df.empty:
        return Response({"error": "No movies in the database"}, status=404)

    # Find numeric row index of the selected movie
    try:
        selected_idx = movies_df.index[movies_df['movieId'] == int(index)][0]
    except IndexError:
        return Response({"error": f"Movie with movieId {index} not found"}, status=404)

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(movies_df['combined_features'].fillna(''))

    # Get recommendations
    recommended_movies = recommend_movies_euclidean(
        selected_idx,
        movies_df,
        n_recommendations=n_recommendations,
        tfidf_matrix=tfidf_matrix
    )
    

    # Return as list of dicts
    return Response(recommended_movies["movieId"].tolist())
