from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from recommendations.models import Movie
import pandas as pd
import numpy as np

def recommend_movies_euclidean(selected_idx, df, n_recommendations=3, tfidf_matrix=None):
    try:
        # Convert the selected movie vector to dense 1D array
        selected_vector = tfidf_matrix[selected_idx, :].toarray()  # shape (1, n_features)
        
        # Compute Euclidean distances from selected movie to all movies
        distances = cdist(selected_vector, tfidf_matrix.toarray(), metric='euclidean').flatten()
        
        # Exclude the selected movie itself
        distances[selected_idx] = np.inf
        
        # Get indices of top N closest movies
        recommended_indices = distances.argsort()[:n_recommendations]
        
        # Return only movieId as a list
        recommended_movie_ids = df.iloc[recommended_indices]["movieId"].tolist()
        return recommended_movie_ids

    except Exception as e:
        # Log the full traceback for Render
        import traceback
        print("Error in recommend_movies_euclidean:", str(e))
        traceback.print_exc()
        return []

@api_view(['GET'])
def index(request, index=0, n_recommendations=3):
    try:
        # Load movies from DB into DataFrame
        queryset = Movie.objects.all().values()
        movies_df = pd.DataFrame(list(queryset))
        
        if movies_df.empty:
            return Response({"error": "No movies found in the database"}, status=500)

        # Find the row index of the selected movie
        selected_idx = movies_df.index[movies_df['movieId'] == int(index)].tolist()
        if not selected_idx:
            return Response({"error": f"MovieId {index} not found"}, status=404)
        selected_idx = selected_idx[0]

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(movies_df['combined_features'])

        # Get recommendations
        recommendations = recommend_movies_euclidean(
            selected_idx,
            movies_df,
            n_recommendations=n_recommendations,
            tfidf_matrix=tfidf_matrix
        )

        return Response(recommendations)

    except Exception as e:
        import traceback
        print("Error in index view:", str(e))
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
