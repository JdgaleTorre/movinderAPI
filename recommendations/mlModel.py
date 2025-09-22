# recommendations/ml_model.py
import tensorflow as tf
import os
from movinderAPI import settings

# Use absolute path — Render typically mounts code at /opt/render/project/src
model_path = os.path.join(settings.BASE_DIR, "Models", "HybridModel.h5")    


    
print("🔄 Loading hybrid model...")
hybrid_model = tf.keras.models.load_model(model_path)
print("✅ Hybrid model loaded once at startup")
