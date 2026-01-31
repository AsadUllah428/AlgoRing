# AlgoRing
A deep learning-powered visual search engine that uses CNN feature extraction (VGG16) to find exact and similar ring images from a dataset. Upload any ring image to discover matches instantly.


# 💍 RingFinder AI - Visual Ring Search Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.10+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

<p align="center">
  <b>A deep learning-powered visual search engine for finding exact and similar ring images using CNN feature extraction</b>
</p>

---

## 📖 Description

**RingFinder AI** is an intelligent image search system that uses Convolutional Neural Networks (CNN) to identify and find similar ring images from a dataset. Upload any ring image, and the system will:

- 🎯 **Find Exact Matches** - Identifies if the exact ring exists in the database
- ✨ **Discover Similar Rings** - Returns visually similar rings ranked by similarity percentage
- ⚡ **Real-time Results** - Get instant results with our optimized feature extraction pipeline

Built with VGG16 transfer learning, the system extracts deep visual features from ring images, enabling accurate visual similarity search without requiring manual tagging or categorization.

---

## 🌟 Features

- **Deep Learning Powered** - Uses VGG16 pre-trained on ImageNet for robust feature extraction
- **Exact Match Detection** - Perceptual hashing algorithm for finding identical images
- **Similarity Search** - Cosine similarity-based ranking of similar images
- **Modern Web Interface** - Drag-and-drop image upload with real-time preview
- **Responsive Design** - Works on desktop and mobile devices
- **Fast Processing** - Pre-computed feature vectors for instant search results



---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AsadUllah428/AlgoRing.git
  
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   - Place original ring images in the `ring/` folder
   - Place augmented images in the `ring_augmented/` folder (optional)

5. **Train the model**
   ```bash
   # Option 1: Run the Jupyter notebook (recommended for visualization)
   jupyter notebook train_model.ipynb
   
   # Option 2: Run the Python script
   python train_model.py
   ```

6. **Start the web server**
   ```bash
   python app.py
   ```

7. **Open your browser**
   ```
   http://localhost:5000
   ```

---

## 📁 Project Structure

```
AlgoRing/
│
├── ring/                    # Original ring images dataset
├── ring_augmented/          # Augmented images for better training
├── templates/
│   └── index.html          # Web interface
│
├── train_model.ipynb       # Training notebook with visualizations
├── train_model.py          # Training script (command-line version)
├── app.py                  # Flask backend server
│
├── ring_model.h5           # Trained feature extraction model
├── ring_database.pkl       # Pre-computed features database
│
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## 🔧 How It Works

### 1. Feature Extraction
The system uses **VGG16**, a pre-trained CNN, to extract 512-dimensional feature vectors from each ring image. These features capture visual characteristics like shape, color, texture, and patterns.

### 2. Database Building
During training, features are extracted from all images and stored in a database along with perceptual hashes for exact matching.

### 3. Similarity Search
When a user uploads an image:
1. Features are extracted from the uploaded image
2. Perceptual hash is computed for exact match detection
3. Cosine similarity is calculated against all database images
4. Results are ranked and returned

### 4. Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Upload    │────▶│   VGG16     │────▶│  Feature    │
│   Image     │     │  Extractor  │     │  Vector     │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Display   │◀────│   Rank by   │◀────│   Cosine    │
│   Results   │     │  Similarity │     │  Similarity │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 🚀 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/search` | POST | Upload image and search for matches |
| `/images/<path>` | GET | Serve dataset images |
| `/health` | GET | Health check endpoint |
| `/stats` | GET | Database statistics |

---

## 📊 Technologies Used

- **Deep Learning**: TensorFlow, Keras, VGG16
- **Backend**: Flask, Flask-CORS
- **Data Processing**: NumPy, Pillow, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Matplotlib, t-SNE

---

## 🔮 Future Improvements

- [ ] Add support for ring classification (engagement, wedding, fashion, etc.)
- [ ] Implement image cropping for better ring detection
- [ ] Add more CNN architectures (ResNet, EfficientNet)
- [ ] Deploy to cloud (AWS, Azure, GCP)
- [ ] Add user accounts and favorites
- [ ] Mobile app development

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---



## 👨‍💻 Author

**Your Name**
- GitHub: [Asad Munir](https://github.com/AsadUllah428)
- LinkedIn: [Asad Munir](https://www.linkedin.com/in/asad-munir-432671168/)

---

## 🙏 Acknowledgments

- VGG16 model by Visual Geometry Group, University of Oxford
- TensorFlow and Keras teams for the amazing deep learning framework
- Flask team for the lightweight web framework

---

<p align="center">
  Made with ❤️ and Python
</p>
