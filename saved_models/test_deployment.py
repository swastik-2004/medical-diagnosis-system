
# Quick test script for deployed ensemble model
import pickle
import pandas as pd
import numpy as np
import os

# Load configuration
with open("saved_models/ensemble_config_complete.pkl", "rb") as f:
    config = pickle.load(f)

print("Ensemble Model Configuration:")
print(f"Version: {config['version']}")
print(f"Created: {config['created_at']}")
print(f"Description: {config['description']}")
print(f"Number of disease classes: {len(config['label_names'])}")

# Test basic functionality
print("\nTesting basic functionality...")
try:
    # Test heart prediction
    heart_df = pd.read_csv("pre_heart.csv").iloc[[0]]
    heart_probs = predict_heart(heart_df)
    print(f"✅ Heart prediction: {heart_probs.shape}")
    
    # Test symptom prediction  
    symptom_probs = predict_symptoms(["chest pain"])
    print(f"✅ Symptom prediction: {symptom_probs.shape}")
    
    print("✅ Basic functionality test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
