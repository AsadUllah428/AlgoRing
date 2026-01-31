"""
Ring Image Feature Extraction and Model Training
This script uses a pre-trained VGG16 CNN to extract features from ring images
for exact matching and similarity search.
"""

import os
import numpy as np
import pickle
from PIL import Image
import hashlib
from tqdm import tqdm

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = 'ring'  # Original images folder
AUGMENTED_DIR = 'ring_augmented'  # Augmented images folder

def create_feature_extractor():
    """
    Create a feature extraction model using VGG16 pre-trained on ImageNet.
    We remove the top classification layers and use the feature vector from
    the last convolutional block.
    """
    print("Loading VGG16 model...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Create a model that outputs features from the last conv layer
    # Using GlobalAveragePooling to get a fixed-size feature vector
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    feature_extractor = Model(inputs=base_model.input, outputs=x)
    
    print(f"Feature extractor created. Output shape: {feature_extractor.output_shape}")
    return feature_extractor

def load_and_preprocess_image(img_path, target_size=IMAGE_SIZE):
    """
    Load an image and preprocess it for VGG16.
    """
    try:
        img = keras_image.load_img(img_path, target_size=target_size)
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def compute_image_hash(img_path):
    """
    Compute a perceptual hash of an image for exact matching.
    Uses average hash (aHash) algorithm.
    """
    try:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((16, 16), Image.Resampling.LANCZOS)
        pixels = np.array(img).flatten()
        avg = pixels.mean()
        hash_bits = ''.join(['1' if p > avg else '0' for p in pixels])
        return hash_bits
    except Exception as e:
        print(f"Error computing hash for {img_path}: {e}")
        return None

def extract_features_batch(model, image_paths, batch_size=BATCH_SIZE):
    """
    Extract features from a batch of images.
    """
    features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = load_and_preprocess_image(path)
            if img is not None:
                batch_images.append(img[0])
        
        if batch_images:
            batch_array = np.array(batch_images)
            batch_features = model.predict(batch_array, verbose=0)
            features.extend(batch_features)
    
    return np.array(features)

def build_database(model, dataset_dir, augmented_dir=None):
    """
    Build a database of image features and hashes for all images in the dataset.
    """
    database = {
        'image_paths': [],
        'features': [],
        'hashes': [],
        'original_paths': []  # Track which images are originals
    }
    
    # Process original images
    print("\nProcessing original images...")
    original_images = []
    if os.path.exists(dataset_dir):
        for filename in os.listdir(dataset_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                img_path = os.path.join(dataset_dir, filename)
                original_images.append(img_path)
                database['original_paths'].append(img_path)
    
    print(f"Found {len(original_images)} original images")
    
    # Process augmented images
    augmented_images = []
    if augmented_dir and os.path.exists(augmented_dir):
        print("\nProcessing augmented images...")
        for filename in os.listdir(augmented_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                img_path = os.path.join(augmented_dir, filename)
                augmented_images.append(img_path)
                # Also add originals from augmented folder
                if '_original' in filename:
                    database['original_paths'].append(img_path)
    
    print(f"Found {len(augmented_images)} augmented images")
    
    # Combine all images
    all_images = original_images + augmented_images
    database['image_paths'] = all_images
    
    # Extract features
    print("\nExtracting features from all images...")
    for i, img_path in enumerate(tqdm(all_images, desc="Extracting features")):
        # Load and preprocess image
        img = load_and_preprocess_image(img_path)
        if img is not None:
            # Extract features
            feature = model.predict(img, verbose=0)[0]
            database['features'].append(feature)
            
            # Compute hash
            img_hash = compute_image_hash(img_path)
            database['hashes'].append(img_hash)
        else:
            # Use zero features and empty hash for failed images
            database['features'].append(np.zeros(512))
            database['hashes'].append(None)
    
    database['features'] = np.array(database['features'])
    
    print(f"\nDatabase built with {len(database['image_paths'])} images")
    print(f"Feature matrix shape: {database['features'].shape}")
    
    return database

def normalize_features(features):
    """
    L2 normalize feature vectors for cosine similarity computation.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return features / norms

def save_model_and_database(model, database, model_path='ring_model.h5', db_path='ring_database.pkl'):
    """
    Save the feature extraction model and the image database.
    """
    # Save the Keras model
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    
    # Normalize features for similarity search
    database['features_normalized'] = normalize_features(database['features'])
    
    # Save the database
    print(f"Saving database to {db_path}...")
    with open(db_path, 'wb') as f:
        pickle.dump(database, f)
    
    print("Model and database saved successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total images indexed: {len(database['image_paths'])}")
    print(f"Original images: {len(database['original_paths'])}")
    print(f"Feature vector dimension: {database['features'].shape[1]}")
    print(f"Model file: {model_path}")
    print(f"Database file: {db_path}")
    print("="*50)

def main():
    """
    Main training pipeline.
    """
    print("="*50)
    print("RING IMAGE FEATURE EXTRACTION SYSTEM")
    print("="*50)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
    else:
        print("No GPU found, using CPU")
    
    # Create feature extractor
    model = create_feature_extractor()
    
    # Get the script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, DATASET_DIR)
    augmented_dir = os.path.join(script_dir, AUGMENTED_DIR)
    
    # Build database
    database = build_database(model, dataset_dir, augmented_dir)
    
    # Save everything
    model_path = os.path.join(script_dir, 'ring_model.h5')
    db_path = os.path.join(script_dir, 'ring_database.pkl')
    save_model_and_database(model, database, model_path, db_path)
    
    print("\nTraining complete! You can now run the Flask app to search for rings.")

if __name__ == "__main__":
    main()
