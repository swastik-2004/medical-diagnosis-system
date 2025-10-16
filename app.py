from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import base64
from io import BytesIO
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Global variables for models
heart_model = None
heart_scaler = None
heart_feature_cols = None
symptom_model = None
tfidf_vectorizer = None
xray_state_dict = None
label_names = []

# Define CNN architecture for X-ray model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def load_models():
    """Load all saved models and components"""
    global heart_model, heart_scaler, heart_feature_cols, symptom_model, tfidf_vectorizer, xray_state_dict, label_names
    
    try:
        # Load ensemble configuration
        with open("saved_models/ensemble_config_complete.pkl", "rb") as f:
            ensemble_config = pickle.load(f)
        label_names = ensemble_config.get('label_names', [])
        print(f"Loaded {len(label_names)} disease labels")
        
        # Load heart disease model and components
        with open("heart_disease_model.pkl", "rb") as f:
            heart_model = pickle.load(f)
        print("Heart disease model loaded")
        
        with open("heart_scaler_final.pkl", "rb") as f:
            heart_scaler = pickle.load(f)
        print("Heart scaler loaded")
        
        with open("heart_feature_columns.pkl", "rb") as f:
            heart_feature_cols = pickle.load(f)
        print("Heart feature columns loaded")
        
        # Load NLP model and vectorizer
        try:
            with open("symptos2disease_model_fixed.pkl", "rb") as f:
                symptom_model = pickle.load(f)
            with open("tfidf_vectorizer_fixed.pkl", "rb") as f:
                tfidf_vectorizer = pickle.load(f)
            print("Symptom model and vectorizer loaded (fixed versions)")
        except FileNotFoundError:
            with open("symptos2disease_model.pkl", "rb") as f:
                symptom_model = pickle.load(f)
            with open("tfidf_vectorizer.pkl", "rb") as f:
                tfidf_vectorizer = pickle.load(f)
            print("Symptom model and vectorizer loaded (original versions)")
        
        # Load X-ray model
        xray_state_dict = torch.load("chest_xray_cnn.pth", map_location='cpu')
        print("X-ray model loaded")
        
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def preprocess_text(text):
    """Preprocess text for NLP model"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def predict_symptoms(symptom_texts):
    """Predict disease from symptom text"""
    if symptom_model is None or tfidf_vectorizer is None:
        return np.array([[0.0] * len(label_names)])
    
    processed_texts = [preprocess_text(text) for text in symptom_texts]
    X_transformed = tfidf_vectorizer.transform(processed_texts)
    probabilities = symptom_model.predict_proba(X_transformed)
    return probabilities

def predict_heart(heart_data):
    """Predict heart disease from input data"""
    if heart_model is None or heart_scaler is None or heart_feature_cols is None:
        return np.array([[0.5, 0.5]])
    
    # Convert input data to DataFrame
    if isinstance(heart_data, dict):
        df = pd.DataFrame([heart_data])
    else:
        df = heart_data.copy()
    
    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'id', 'num', 'label']
    X_heart = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    
    # Align with training columns
    for col in heart_feature_cols:
        if col not in X_heart.columns:
            X_heart[col] = 0
    X_heart = X_heart[heart_feature_cols]
    
    # Apply scaling
    X_heart = heart_scaler.transform(X_heart)
    
    # Predict
    probs = heart_model.predict_proba(X_heart)
    return probs

def predict_xray(image_path):
    """Predict pneumonia from chest X-ray image"""
    if xray_state_dict is None:
        return np.array([[0.5, 0.5]])
        
    try:
        # Create model instance
        model = SimpleCNN(num_classes=2)
        model.load_state_dict(xray_state_dict)
        model.eval()
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
        return probabilities.numpy()
        
    except Exception as e:
        print(f"Error in X-ray prediction: {e}")
        return np.array([[0.5, 0.5]])

def map_binary_to_multiclass(symptom_probs, heart_probs, xray_probs, label_names):
    """Map binary outputs to multiclass probabilities"""
    final_probs = symptom_probs.copy()
    
    heart_disease_prob = heart_probs[0, 1] if heart_probs.size > 0 else 0
    pneumonia_prob = xray_probs[0, 1] if xray_probs.size > 0 else 0
    
    # Add contributions to mapped diseases
    if "Hypertension" in label_names:
        idx = label_names.index("Hypertension")
        final_probs[0, idx] += 0.3 * heart_disease_prob
    
    if "Pneumonia" in label_names:
        idx = label_names.index("Pneumonia")
        final_probs[0, idx] += 0.4 * pneumonia_prob
    
    # Normalize
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
    return final_probs

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', diseases=label_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        symptom_text = request.form.get('symptoms', '')
        xray_file = request.files.get('xray_image')
        
        # Get heart disease parameters
        heart_params = {}
        for param in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
            value = request.form.get(param)
            if value:
                try:
                    heart_params[param] = float(value)
                except ValueError:
                    heart_params[param] = 0.0
            else:
                heart_params[param] = 0.0
        
        # Make predictions
        results = {}
        
        # Symptom prediction
        if symptom_text:
            symptom_probs = predict_symptoms([symptom_text])
            results['symptom_prediction'] = {
                'probabilities': symptom_probs[0].tolist(),
                'top_diseases': []
            }
            
            # Get top 5 predictions
            top_indices = np.argsort(symptom_probs[0])[-5:][::-1]
            for idx in top_indices:
                results['symptom_prediction']['top_diseases'].append({
                    'disease': label_names[idx],
                    'probability': float(symptom_probs[0][idx])
                })
        
        # Heart disease prediction
        if heart_params:
            heart_probs = predict_heart(heart_params)
            results['heart_prediction'] = {
                'no_disease': float(heart_probs[0][0]),
                'disease': float(heart_probs[0][1])
            }
        
        # X-ray prediction
        if xray_file and xray_file.filename:
            # Save uploaded file temporarily
            temp_path = f"temp_{xray_file.filename}"
            xray_file.save(temp_path)
            
            try:
                xray_probs = predict_xray(temp_path)
                results['xray_prediction'] = {
                    'normal': float(xray_probs[0][0]),
                    'pneumonia': float(xray_probs[0][1])
                }
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Combined prediction if we have multiple inputs
        if len(results) > 1:
            symptom_probs = results.get('symptom_prediction', {}).get('probabilities', [0.0] * len(label_names))
            if not symptom_probs:
                symptom_probs = [0.0] * len(label_names)
            symptom_probs = np.array([symptom_probs])
            
            heart_probs = np.array([[results.get('heart_prediction', {}).get('no_disease', 0.5),
                                   results.get('heart_prediction', {}).get('disease', 0.5)]])
            
            xray_probs = np.array([[results.get('xray_prediction', {}).get('normal', 0.5),
                                  results.get('xray_prediction', {}).get('pneumonia', 0.5)]])
            
            combined_probs = map_binary_to_multiclass(symptom_probs, heart_probs, xray_probs, label_names)
            
            results['combined_prediction'] = {
                'probabilities': combined_probs[0].tolist(),
                'top_diseases': []
            }
            
            # Get top 5 combined predictions
            top_indices = np.argsort(combined_probs[0])[-5:][::-1]
            for idx in top_indices:
                results['combined_prediction']['top_diseases'].append({
                    'disease': label_names[idx],
                    'probability': float(combined_probs[0][idx])
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/diseases')
def get_diseases():
    """API endpoint to get list of diseases"""
    return jsonify({'diseases': label_names})

if __name__ == '__main__':
    print("Loading models...")
    if load_models():
        print("All models loaded successfully!")
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check your model files.")
