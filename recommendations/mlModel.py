# recommendations/ml_model.py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from movinderAPI import settings
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization
from keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from recommendations.models import Movie, MovieVote, User
import pandas as pd


# Use absolute path — Render typically mounts code at /opt/render/project/src
model_path = os.path.join(settings.BASE_DIR, "Models", "HybridModel.keras")    


    
print("🔄 Loading hybrid model...")
hybrid_model = tf.keras.models.load_model(model_path)
print("✅ Hybrid model loaded once at startup")


def train_hybrid_model():
    print("🚀 Starting hybrid model training...")

    # 1️⃣ Pull combined features
    qs = Movie.objects.all().values('id','combined_features')
    votes = MovieVote.objects.all().values('movieId')
    combined_features_list = [q['combined_features'] for vote in votes for q in qs if q['id'] == vote['movieId']]
    print(f"📊 Pulled {len(combined_features_list)} combined features from DB, votes {len(votes)}")

    # 2️⃣ Text Vectorization
    max_tokens = 5000
    max_length = 100

    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_length
    )

    try:
        print("🔄 Adapting TextVectorization...")
        vectorizer.adapt(combined_features_list)
        sequences = vectorizer(combined_features_list)
        features_padded_Hybrid_Third_Structure = sequences.numpy()
        print(f"✅ Vectorized features shape: {features_padded_Hybrid_Third_Structure.shape}")
    except Exception as e:
        print("❌ Error in vectorizing text:", e)
        return []

    # 3️⃣ Pull user/movie/vote data
    print("📥 Fetching MovieVote rows with IDs & votes...")
    X_qs = MovieVote.objects.all().values('createdById', 'movieId', 'vote')
    X_list = list(X_qs)
    print(f"📊 Pulled {len(X_list)} rows of MovieVotes")
    
    # Fetch users and movies
    users = list(User.objects.all().values('id'))
    movies = list(Movie.objects.all().values('id'))
    
    user_map = {user['id']: i for i, user in enumerate(users)}
    movie_map = {movie['id']: i for i, movie in enumerate(movies)}
    
    MAX_RATING = 5

    try:
        # Convert to NumPy arrays
        created_by_ids = np.array([user_map[row['createdById']] for row in X_list])
        movie_ids = np.array([movie_map[row['movieId']] for row in X_list])
        votes = np.array([row['vote'] / MAX_RATING for row in X_list])

        print(f"🔢 created_by_ids shape: {created_by_ids.shape}")
        print(f"🔢 movie_ids shape: {movie_ids.shape}")
        print(f"🔢 votes shape: {votes.shape}")
        print(f"🔢 features shape: {features_padded_Hybrid_Third_Structure.shape}")

        # Concatenate
        # X = np.column_stack([created_by_ids, movie_ids, votes, features_padded...])
        X_numeric = np.column_stack([created_by_ids, movie_ids, votes])
        X = np.hstack([X_numeric, features_padded_Hybrid_Third_Structure])
        y = votes / 5.0
        print(f"✅ Final X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print("❌ Error assembling X and y arrays:", e)
        return []

    # 4️⃣ Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    except Exception as e:
        print("❌ Error splitting train/test:", e)
        return []

    # 5️⃣ Train model
    try:
        print("Compiling Hybrid model...")
        # Define user and movie input layers
        num_users = len(user_map)
        num_movies = max(movie_map.values()) + 1
        embedding_dim = 100

        # User embedding
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, name='user_embedding')(user_input)

        # Movie embedding
        movie_input = Input(shape=(1,), name='movie_input')
        movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim, name='movie_embedding')(movie_input)

        # Rating embedding
        rating_input = Input(shape=(1,), name='rating_input')
        rating_embedding = Embedding(input_dim=num_movies, output_dim=embedding_dim, name='rating_embedding')(rating_input)

        # --- Content-Based Part ---
        content_input = Input(shape=(max_length,), name="content_features")
        x = Dense(128, activation="elu")(content_input)
        x = Dense(64, activation="elu")(x)
        content_embedding = Dense(32, activation="elu", name='content_embedding')(x)
        content_flatten = Flatten()(content_embedding)

        # --- Concatenate User, Movie, and Rating Embeddings ---
        concat_embeddings = Concatenate()([user_embedding, movie_embedding, rating_embedding])
        concat_flatten = Flatten()(concat_embeddings)  # <-- instead of LSTM

        # --- Hybrid Model ---
        combined_embeddings = Concatenate()([concat_flatten, content_flatten])

        x = Dense(512, activation='elu')(combined_embeddings)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='elu')(x)
        x = BatchNormalization()(x)

        # Output layer
        output = Dense(1, activation='sigmoid')(x)

        # Define and compile the hybrid model
        hybrid_model = Model(inputs=[user_input, movie_input, rating_input, content_input],
                                outputs=output, name="HybridNN")
        hybrid_model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

        hybrid_model.summary()


        print("🚀 Training hybrid model...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = hybrid_model.fit(
            [X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3:]],
            y_train,
            epochs=100,
            batch_size=128,
            callbacks=[early_stopping],
            validation_data=([X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3:]], y_test),
            verbose=1  # ensure logs print in Hugging Face console
        )
        print("✅ Training complete.")
    except Exception as e:
        print("❌ Error during model training:", e)
        return []

    # 6️⃣ Save model
    try:
        hybrid_model.save(model_path)
        print(f"💾 Model saved to {model_path}")
    except Exception as e:
        print("❌ Error saving model:", e)
        return []

    print("🎉 Hybrid model retrained and saved successfully.")
    return True