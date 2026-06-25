# RiskShield AI: Financial Fraud Detection Platform & Dashboard

Welcome to **RiskShield AI**, a comprehensive, enterprise-grade data science and machine learning platform built to identify suspicious transactions, map fraudulent networks, and monitor real-time banking logs.

This repository serves as a self-contained, internship-ready portfolio project featuring a full-scale ETL pipeline, nine machine learning models, graph network relationship visualization, and an interactive Streamlit dashboard.

---

## 📂 Project Directory Structure

Everything needed to run, review, and deploy the project is contained within this folder:

```
Financial_Fraud_Detection_Project/
├── dashboard/               # Streamlit application UI
│   └── app.py               # Main dashboard script (Overview, EDA, Graph, Arena, Simulator)
├── data/                    # Storage for databases and datasets
│   ├── datasets/            # Raw datasets (.csv)
│   │   ├── synthetic_fraud_dataset1.csv       # Main 50k transaction synthetic dataset
│   │   ├── fraud_detection_dataset.csv        # 100k transaction credit card dataset
│   │   └── financial_fraud_detection_dataset.csv # 5k transaction customer dataset
│   ├── fraud_detection.db   # Local SQLite database containing ETL tables
│   └── fraud_alerts.json    # JSON log files tracking generated alerts
├── models/                  # Serialized pipelines and model binaries
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lightgbm.pkl
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── autoencoder.pkl
│   ├── ensemble_model.pkl
│   ├── scaler.pkl           # MinMaxScaler for normalized prediction
│   ├── encoders.pkl         # LabelEncoders for categorical columns
│   ├── feature_names.json   # Features list used for training alignment
│   └── metrics.json         # Performance metrics saved from training
├── notebooks/               # Offline model development
│   └── predict_fraud.py     # Batch training and validation pipeline script
├── utils/                   # Shared utility modules
│   └── helpers.py           # Preprocessing, database queries, graph construction, alerting functions
└── requirements.txt         # Project dependencies
```

---

## ⚡ Key Architecture & Features

### 1. Data Acquisition & SQLite ETL Pipeline
Rather than reading flat files on every run, the platform implements an automated ETL pipeline:
* **Extract:** Pulls raw records from three distinct financial transaction datasets.
* **Transform:** Standardizes snake_case naming conventions, parses dates, handles NaN fields, and performs log transformations on transaction volumes.
* **Load:** Loads the standardized datasets into three distinct tables (`synthetic_transactions`, `credit_card_transactions`, and `financial_transactions`) within a local SQLite database (`data/fraud_detection.db`).
* **Console:** An interactive SQL console is provided inside the dashboard to execute query tests directly against SQLite.

### 2. Supervised & Unsupervised Machine Learning Models
RiskShield AI trains and evaluates nine distinct classification and anomaly detection models:
* **Supervised Models:**
  1. *Logistic Regression:* The baseline linear model.
  2. *Decision Tree:* Structural regularization is applied (`max_depth=8`, `min_samples_split=5`, `min_samples_leaf=3`) to prevent tree overfitting.
  3. *Random Forest:* Regularized ensemble bagging model (`n_estimators=100`, `max_depth=10`).
  4. *XGBoost:* Highly optimized Gradient Boosting classifier.
  5. *LightGBM:* Fast, histogram-based gradient boosting.
* **Unsupervised Models:**
  6. *Isolation Forest:* Isolates transactions by partitioning boundaries; flags anomalies based on short tree paths.
  7. *One-Class SVM:* Learns a decision boundary representing legitimate transactions, classifying deviations as fraud.
  8. *Autoencoder (Deep Learning):* Built using an `MLPRegressor` with a bottleneck compression layer (`8 -> 4 -> 8` neurons). Trained strictly on legitimate data to learn normal transaction behavior; anomaly scores are computed using reconstruction Mean Squared Error (MSE).
* **Ensemble Voting:**
  9. *Voting Classifier:* Aggregates prediction probabilities from Logistic Regression, Random Forest, and XGBoost using soft voting.

### 3. Class Balancing (SMOTE) & Normalization
To prevent models from prioritizing the majority (legitimate) class, a **SMOTE (Synthetic Minority Over-sampling Technique)** pipeline is applied to the training set to balance fraud examples. Features are scaled between `[0, 1]` using `MinMaxScaler` for distance-based estimators.

### 4. Graph-Based Network Analysis
Coordinates and locations are mapped to a transactional network using `networkx`:
* Nodes represent transactions and entities (locations, merchants, users).
* Edges represent sharing associations.
* Red nodes highlight flagged fraudulent transactions, exposing structured **money laundering rings** and suspicious multi-hop paths.
* Displays live network density, node degree, and clustering coefficients.

### 5. Real-Time Streaming Simulator & Alerts
Simulates live bank transaction ingestion:
* Pulls live logs randomly from the SQLite database.
* Performs inference through the selected machine learning model in real-time.
* Renders a live terminal feed showing active alerts.
* Triggers visual browser alert frames and supports a pipeline that logs anomalies to a local file (`data/fraud_alerts.json`) and sends SMTP emails.

---

## 🚀 How to Set Up and Run

### Prerequisites
Make sure you have **Python 3.10+** installed.

### 1. Install Dependencies
Navigate to the project directory and install the necessary libraries:
```bash
pip install -r requirements.txt
```

### 2. Pre-train Models (Optional)
The project comes with pre-trained models. However, if you want to run the training pipeline yourself:
```bash
python notebooks/predict_fraud.py
```
This script will preprocess the dataset, apply SMOTE, train all 9 models, and export metrics to `models/metrics.json`.

### 3. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run dashboard/app.py
```
A browser window will automatically launch at `http://localhost:8501`.

---

## 🛡️ Executive Report & Insights

### Key Takeaways
1. **Unsupervised vs. Supervised:** Supervised models (e.g. XGBoost, Random Forest) achieve higher class accuracy on balanced test splits, but unsupervised models (Autoencoder reconstruction, Isolation Forest) are far more robust at detecting zero-day fraud signatures.
2. **SMOTE Impact:** Implementing SMOTE significantly reduces false negatives (increasing Recall), which is critical for financial institutions where missing a fraudulent transaction is far more expensive than a false alarm.
3. **Graph Networks:** Traditional models look at single transaction rows, but graph-based analysis successfully flags group anomalies where multiple different cards/users execute small transactions at the exact same location or device terminal within minutes.

---
*Developed as a Data Science & Analytics Portfolio project.*