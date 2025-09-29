# recommendations/ml_model.py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from movinderAPI import settings
from keras.callbacks import EarlyStopping
from keras.layers import TextVectorization

from recommendations.models import MovieVote
import pandas as pd


# Use absolute path — Render typically mounts code at /opt/render/project/src
model_path = os.path.join(settings.BASE_DIR, "Models", "HybridModel.keras")    


    
print("🔄 Loading hybrid model...")
hybrid_model = tf.keras.models.load_model(model_path)
print("✅ Hybrid model loaded once at startup")


def train_hybrid_model():
    print("🚀 Starting hybrid model training...")

    # 1️⃣ Pull combined features
    votes_qs = MovieVote.objects.all().values('movie__combined_features')
    combined_features_list = [movie['movie__combined_features'] for movie in votes_qs]
    print(f"📊 Pulled {len(combined_features_list)} combined features from DB")

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

    try:
        # Convert to NumPy arrays
        created_by_ids = np.array([row['createdById'] for row in X_list])
        movie_ids = np.array([row['movieId'] for row in X_list])
        votes = np.array([row['vote'] for row in X_list])

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