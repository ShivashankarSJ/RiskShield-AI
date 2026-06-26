import os
import sys
import json
import numpy as np
import pandas as pd
import pickle

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import (
    preprocess_dataset, train_model, 
    evaluate_model_metrics, save_trained_model, DB_PATH, MODELS_DIR
)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_and_save_all():
    print("Starting Model Training Pipeline...")
    
    # 1. Load synthetic data from database or CSV
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets', 'synthetic_fraud_dataset1.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return
        
    print(f"Loading dataset from: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    
    # 2. Preprocess
    print("Preprocessing data and engineering features...")
    df_proc, features, target, encoders = preprocess_dataset(df_raw, 'synthetic')
    
    # Save encoders
    encoders_path = os.path.join(MODELS_DIR, 'encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print("Encoders saved.")
    
    # Separate features and target
    X = df_proc[features]
    y = df_proc[target]
    
    # Save feature names
    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(features, f)
    print(f"Feature names saved: {features}")
    
    # 3. Train-Test Split (stratified)
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Class Balancing using SMOTE (apply on training set only)
    print("Balancing training set using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Original shape: {X_train.shape}, Balanced shape: {X_train_res.shape}")
    
    # 5. Normalization using MinMaxScaler
    print("Normalizing features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features)
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")
    
    # List of models to train
    model_types = [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'XGBoost',
        'LightGBM',
        'Isolation Forest',
        'One-Class SVM',
        'Autoencoder',
        'Ensemble Model'
    ]
    
    metrics_report = {}
    
    # 6. Train and evaluate each model
    for model_type in model_types:
        print(f"\n--- Training {model_type} ---")
        try:
            model_filename = model_type.lower().replace(' ', '_').replace('-', '_') + '.pkl'
            
            # Train
            # Note: Unsupervised models get unscaled/scaled differently, but for consistency we pass normalized data
            clf = train_model(X_train_scaled_df, y_train_res, model_type=model_type, random_state=42)
            
            # Save
            save_trained_model(clf, model_filename)
            print(f"Saved {model_type} to {model_filename}")
            
            # Evaluate
            print(f"Evaluating {model_type}...")
            metrics = evaluate_model_metrics(clf, X_test_scaled_df, y_test, model_type)
            
            # Log key results
            print(f"Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_score']:.4f} | ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Store in metrics report (excluding raw FPR/TPR to keep size smaller or downsampling them)
            # Let's downsample FPR/TPR to 100 points for ROC curve plotting in Streamlit
            fpr = metrics.pop('fpr')
            tpr = metrics.pop('tpr')
            
            # Downsampling function
            indices = np.linspace(0, len(fpr) - 1, min(100, len(fpr)), dtype=int)
            metrics['fpr_curve'] = [float(fpr[i]) for i in indices]
            metrics['tpr_curve'] = [float(tpr[i]) for i in indices]
            
            metrics_report[model_type] = metrics
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            
    # 7. Save metrics report
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_report, f, indent=4)
    print(f"\nAll models trained! Performance metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_and_save_all()