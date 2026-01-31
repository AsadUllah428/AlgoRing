"""
Ring Image Search Flask Backend
Handles image uploads, processes them through the trained model,
and returns exact matches and similar images.
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
import io

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
MODEL_PATH = 'ring_model.h5'
DATABASE_PATH = 'ring_database.pkl'
IMAGE_SIZE = (224, 224)
TOP_K_SIMILAR = 12  # Number of similar images to return
EXACT_MATCH_THRESHOLD = 0.98  # Similarity threshold for exact match

# Global variables for model and database
feature_extractor = None
database = None

def load_model_and_database():
    """
    Load the trained model and image database.
    """
    global feature_extractor, database
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_PATH)
    db_path = os.path.join(script_dir, DATABASE_PATH)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please run train_model.py first.")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}. Please run train_model.py first.")
    
    print("Loading feature extraction model...")
    feature_extractor = load_model(model_path)
    
    print("Loading image database...")
    with open(db_path, 'rb') as f:
        database = pickle.load(f)
    
    print(f"Loaded database with {len(database['image_paths'])} images")
    print("Model and database loaded successfully!")

def preprocess_image(img):
    """
    Preprocess an image for the VGG16 model.
    """
    # Resize image
    img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to array and preprocess
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array

def compute_image_hash(img):
    """
    Compute perceptual hash for an image.
    """
    # Convert to grayscale and resize
    img_gray = img.convert('L')
    img_small = img_gray.resize((16, 16), Image.Resampling.LANCZOS)
    
    # Compute hash
    pixels = np.array(img_small).flatten()
    avg = pixels.mean()
    hash_bits = ''.join(['1' if p > avg else '0' for p in pixels])
    
    return hash_bits

def hamming_distance(hash1, hash2):
    """
    Compute Hamming distance between two hashes.
    """
    if hash1 is None or hash2 is None:
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def normalize_feature(feature):
    """
    L2 normalize a feature vector.
    """
    norm = np.linalg.norm(feature)
    if norm == 0:
        return feature
    return feature / norm

def find_matches(query_feature, query_hash, top_k=TOP_K_SIMILAR):
    """
    Find exact match and similar images.
    """
    # Normalize query feature
    query_feature_norm = normalize_feature(query_feature)
    
    # Compute similarities using normalized features
    similarities = np.dot(database['features_normalized'], query_feature_norm)
    
    # Find exact match based on hash (low Hamming distance) and high similarity
    exact_match = None
    exact_match_idx = None
    
    for i, db_hash in enumerate(database['hashes']):
        if db_hash is not None and query_hash is not None:
            h_dist = hamming_distance(query_hash, db_hash)
            if h_dist <= 10 and similarities[i] > EXACT_MATCH_THRESHOLD:  # Very similar hash
                exact_match = database['image_paths'][i]
                exact_match_idx = i
                break
    
    # If no hash match, check for very high similarity
    if exact_match is None:
        max_sim_idx = np.argmax(similarities)
        if similarities[max_sim_idx] > EXACT_MATCH_THRESHOLD:
            exact_match = database['image_paths'][max_sim_idx]
            exact_match_idx = max_sim_idx
    
    # Get top-k similar images (excluding exact match)
    sorted_indices = np.argsort(similarities)[::-1]
    
    similar_images = []
    for idx in sorted_indices:
        if idx == exact_match_idx:
            continue
        if len(similar_images) >= top_k:
            break
        
        similar_images.append({
            'path': database['image_paths'][idx],
            'similarity': float(similarities[idx]) * 100  # Convert to percentage
        })
    
    return exact_match, similar_images

@app.route('/')
def index():
    """
    Serve the main HTML page.
    """
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """
    Search for matching and similar ring images.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Preprocess for model
        img_preprocessed = preprocess_image(img)
        
        # Extract features
        query_feature = feature_extractor.predict(img_preprocessed, verbose=0)[0]
        
        # Compute hash
        query_hash = compute_image_hash(img)
        
        # Find matches
        exact_match, similar_images = find_matches(query_feature, query_hash)
        
        return jsonify({
            'exact_match': exact_match,
            'similar_images': similar_images
        })
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:image_path>')
def serve_image(image_path):
    """
    Serve images from the dataset.
    """
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Handle both relative and absolute paths
        if os.path.isabs(image_path):
            full_path = image_path
        else:
            full_path = os.path.join(script_dir, image_path)
        
        # Security check - ensure the path is within allowed directories
        allowed_dirs = [
            os.path.join(script_dir, 'ring'),
            os.path.join(script_dir, 'ring_augmented')
        ]
        
        full_path = os.path.normpath(full_path)
        is_allowed = any(full_path.startswith(os.path.normpath(d)) for d in allowed_dirs)
        
        if not is_allowed:
            return jsonify({'error': 'Access denied'}), 403
        
        if not os.path.exists(full_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(full_path, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"Error serving image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': feature_extractor is not None,
        'database_loaded': database is not None,
        'total_images': len(database['image_paths']) if database else 0
    })

@app.route('/stats')
def stats():
    """
    Return database statistics.
    """
    if database is None:
        return jsonify({'error': 'Database not loaded'}), 500
    
    return jsonify({
        'total_images': len(database['image_paths']),
        'original_images': len(database['original_paths']),
        'feature_dimension': database['features'].shape[1] if len(database['features'].shape) > 1 else 0
    })

if __name__ == '__main__':
    print("="*50)
    print("RING IMAGE SEARCH SERVER")
    print("="*50)
    
    try:
        load_model_and_database()
        print("\nStarting Flask server...")
        print("Open http://localhost:5000 in your browser")
        print("="*50)
        app.run(host='0.0.0.0', port=5000, debug=False)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run train_model.py first to train the model and build the database.")
