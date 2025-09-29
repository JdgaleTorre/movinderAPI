import os
from django.conf import settings
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences
from recommendations.models import Movie, User, MovieVote
import pandas as pd
import numpy as np
from .mlModel import hybrid_model, train_hybrid_model  # Ensure this import is correct

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


@api_view(['GET'])
def ping(request):
    return JsonResponse({"message": "pong!"})


def recommend_Hybrid_Third_Structure(user_id, top_n=5):
    print("Starting hybrid recommendation...")
    
    # Fetch users and movies
    users = list(User.objects.all().values('id'))
    movies = list(Movie.objects.all().values('id', 'movieId', 'combined_features'))
    print(f"Total users: {len(users)}, total movies: {len(movies)}")

    if not users or not movies:
        print("No users or movies found in DB!")
        return []

    user_map = {user['id']: i for i, user in enumerate(users)}
    movie_map = {movie['id']: i for i, movie in enumerate(movies)}
    reverse_movie_map = {i: movie['movieId'] for i, movie in enumerate(movies)}

    print("User map and movie map created.")

    # Extract combined_features as a list
    combined_features_list = [movie['combined_features'] for movie in movies]
    # print(f"First 3 combined_features examples: {combined_features_list[:3]}")

    # Use TextVectorization instead of deprecated Tokenizer
    max_tokens = 5000
    max_length = 100

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_length
    )

    try:
        vectorizer.adapt(combined_features_list)
        sequences = vectorizer(combined_features_list)
        features_padded_Hybrid_Third_Structure = sequences.numpy()
        print(f"Vectorized features shape: {features_padded_Hybrid_Third_Structure.shape}")
    except Exception as e:
        print("Error in vectorizing text:", e)
        return []

    # Map user and movie indices
    user_idx = user_map.get(user_id)
    if user_idx is None:
        print(f"User ID {user_id} not found in user_map!")
        return []

    movie_ids = np.array(list(movie_map.values()))
    content_features = features_padded_Hybrid_Third_Structure

    print(f"User index: {user_idx}, number of movies: {len(movie_ids)}")

    # Initialize ratings array
    ratings = np.zeros(len(movie_ids))

    # Collect user ratings
    user_ratings = MovieVote.objects.filter(createdById=user_id).values('movieId', 'vote')
    print(f"Found {user_ratings.count()} votes for user {user_id}")
    for rate in user_ratings:
        movie_index = movie_map.get(rate['movieId'])
        if movie_index is not None:
            ratings[movie_index] = rate['vote']

    print(f"Ratings array sample: {ratings[:10]}")

    # Predict using the hybrid model
    try:
        predictions = hybrid_model.predict([
            np.array([user_idx] * len(movie_ids)),  # user indices
            movie_ids,                             # movie indices
            ratings,                               # existing ratings
            content_features                        # movie content features
        ])
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print("Error during model prediction:", e)
        return []

    # Get top-N recommendations
    top_indices = predictions.flatten().argsort()[-top_n:][::-1]
    recommended_movie_ids = [movie_ids[i] for i in top_indices]

    tmdbIds = [reverse_movie_map.get(movie, None) for movie in recommended_movie_ids]
    print('Recommended TMDB IDs:', tmdbIds)

    return tmdbIds


@api_view(['GET'])
def hybridNeuralNetworkRecomendations(request, userId, n_recommendations=10):
    print(f"API called for userId={userId}, n_recommendations={n_recommendations}")
    try:
        recommendations = recommend_Hybrid_Third_Structure(userId, top_n=n_recommendations)
        print("Returning recommendations:", recommendations)
        return Response(recommendations)
    except Exception as e:
        print("Error in API view:", e)
        return Response({"error": str(e)}, status=500)


@api_view(['GET'])
def trainModel(request):
    try:
        print('Training model initiated via API call')
        train_hybrid_model()
        return Response({"message": "Model training initiated"})
    except Exception as e:
        print("Error in trainModel view:", e)
        return Response({"error": str(e)}, status=500)