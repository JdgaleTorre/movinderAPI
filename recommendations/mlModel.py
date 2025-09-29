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
    combined_features_Second_Structure =  MovieVote.objects.all().values('combined_features')
    combined_features_list = [movie['combined_features'] for movie in combined_features_Second_Structure]
    # Tokenize descriptions
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
    
    # Replace 'combined_features' in X with the padded sequences
    X = MovieVote.objects.all().values('createdById', 'movieId', 'vote')    
    X = np.hstack([X, features_padded_Hybrid_Third_Structure])
    # Define target variable
    y = X['vote'].values/5.0

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # here we will Implement early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # we need to train with early stopping
    history = hybrid_model.fit([X_train[:, 0], X_train[:, 1],X_train[:, 2],X_train[:, 3:]], 
                            y_train, 
                            epochs=100, 
                            batch_size=128, 
                            callbacks=[early_stopping],
                            validation_data=([X_test[:, 0], X_test[:, 1],X_test[:, 2],X_test[:,3:]],y_test)
                            )
    
    # Save the trained model
    hybrid_model.save(model_path)
    print("✅ Hybrid model retrained and saved")