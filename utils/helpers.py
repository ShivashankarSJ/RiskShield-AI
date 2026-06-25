import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
import json
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor  # Used as Autoencoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import networkx as nx

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(DATA_DIR, 'fraud_detection.db')

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. ETL & Database Functions
# ---------------------------------------------------------------------
def init_db():
    """Perform ETL: Read CSVs, clean column names, and load into SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Synthetic dataset
    synth_path = os.path.join(DATASETS_DIR, 'synthetic_fraud_dataset1.csv')
    if os.path.exists(synth_path):
        df_synth = pd.read_csv(synth_path)
        # Basic Clean
        df_synth.columns = [col.strip() for col in df_synth.columns]
        df_synth.to_sql('synthetic_transactions', conn, if_exists='replace', index=False)
        
    # 2. Credit Card dataset (the original fraud_detection_dataset.csv)
    cc_path = os.path.join(DATASETS_DIR, 'fraud_detection_dataset.csv')
    if os.path.exists(cc_path):
        df_cc = pd.read_csv(cc_path)
        df_cc.columns = [col.strip() for col in df_cc.columns]
        df_cc.to_sql('credit_card_transactions', conn, if_exists='replace', index=False)
        
    # 3. Financial dataset
    fin_path = os.path.join(DATASETS_DIR, 'financial_fraud_detection_dataset.csv')
    if os.path.exists(fin_path):
        df_fin = pd.read_csv(fin_path)
        df_fin.columns = [col.strip() for col in df_fin.columns]
        df_fin.to_sql('financial_transactions', conn, if_exists='replace', index=False)
        
    conn.commit()
    conn.close()
    return True

def query_db(query, params=()):
    """Execute a query on the SQLite database and return a DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df

def get_table_row_count(table_name):
    """Get number of records in a table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT count(*) FROM {table_name}")
        count = cursor.fetchone()[0]
    except Exception:
        count = 0
    finally:
        conn.close()
    return count

# Initialize DB on import if not exists
if not os.path.exists(DB_PATH) or get_table_row_count('synthetic_transactions') == 0:
    try:
        init_db()
    except Exception as e:
        print(f"Error initializing DB: {e}")

# ---------------------------------------------------------------------
# 2. Preprocessing & Feature Engineering
# ---------------------------------------------------------------------
def preprocess_dataset(df, dataset_name):
    """Preprocess a given dataset, returning features and target.
    Handles encoding, missing values, feature engineering, and standardizes names.
    Returns: df_processed, features_list, target_col
    """
    df = df.copy()
    
    if dataset_name == 'synthetic':
        # Target column: Fraud_Label
        target_col = 'is_fraud'
        df = df.rename(columns={'Fraud_Label': 'is_fraud'})
        
        # Categorical columns
        categorical_cols = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Card_Type']
        # Label encode
        encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col.lower()}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                
        # Parse date and extract features
        if 'Date' in df.columns:
            df['parsed_date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['month'] = df['parsed_date'].dt.month.fillna(1).astype(int)
            df['day'] = df['parsed_date'].dt.day.fillna(1).astype(int)
            df['day_of_week'] = df['parsed_date'].dt.dayofweek.fillna(0).astype(int)
        else:
            df['month'] = 1
            df['day'] = 1
            df['day_of_week'] = 0
            
        # Log-transform amount
        if 'Transaction_Amount' in df.columns:
            df['amount_log'] = np.log1p(df['Transaction_Amount'])
            
        # Fill numeric NaNs
        numeric_cols = ['Transaction_Amount', 'Account_Balance', 'Previous_Fraudulent_Activity', 
                        'Daily_Transaction_Count', 'Card_Age', 'amount_log', 'month', 'day', 'day_of_week']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
                
        # Features to keep for ML
        feature_cols = [
            'Transaction_Amount', 'Account_Balance', 'Previous_Fraudulent_Activity', 
            'Daily_Transaction_Count', 'Card_Age', 'amount_log', 'day_of_week',
            'transaction_type_encoded', 'device_type_encoded', 'location_encoded', 
            'merchant_category_encoded', 'card_type_encoded'
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]
        
    elif dataset_name == 'credit_card':
        target_col = 'is_fraud'
        df = df.rename(columns={'IsFraud': 'is_fraud', 'Amount': 'amount', 'Location': 'location', 'TransactionType': 'transaction_type'})
        
        categorical_cols = ['transaction_type', 'location']
        encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col.lower()}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                
        if 'TransactionDate' in df.columns:
            df['parsed_date'] = pd.to_datetime(df['TransactionDate'], format='%d-%m-%Y', errors='coerce')
            df['hour'] = df['parsed_date'].dt.hour.fillna(0).astype(int)
            df['day_of_week'] = df['parsed_date'].dt.dayofweek.fillna(0).astype(int)
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
            
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            
        numeric_cols = ['amount', 'hour', 'day_of_week', 'amount_log']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
                
        feature_cols = ['amount', 'hour', 'day_of_week', 'amount_log', 'transaction_type_encoded', 'location_encoded']
        feature_cols = [col for col in feature_cols if col in df.columns]
        
    else:  # financial
        target_col = 'is_fraud'
        df = df.rename(columns={'Fraudulent': 'is_fraud', 'Transaction_Amount': 'amount'})
        
        categorical_cols = ['Merchant_Category', 'Payment_Method', 'Device_Type', 'Location']
        encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col.lower()}_encoded'] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                
        if 'Suspicious_Keyword' in df.columns:
            df['suspicious_keyword_encoded'] = df['Suspicious_Keyword'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
            
        if 'Transaction_Date' in df.columns:
            df['parsed_date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            df['hour'] = df['parsed_date'].dt.hour.fillna(0).astype(int)
            df['day_of_week'] = df['parsed_date'].dt.dayofweek.fillna(0).astype(int)
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
            
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            
        numeric_cols = ['amount', 'hour', 'day_of_week', 'amount_log', 'Is_International', 
                        'Previous_Transactions', 'Average_Spend', 'Account_Age_Days']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 0)
                
        feature_cols = [
            'amount', 'hour', 'day_of_week', 'amount_log', 'Is_International', 
            'Previous_Transactions', 'Average_Spend', 'Account_Age_Days',
            'suspicious_keyword_encoded', 'merchant_category_encoded', 
            'payment_method_encoded', 'device_type_encoded', 'location_encoded'
        ]
        feature_cols = [col for col in feature_cols if col in df.columns]
        
    # Standardize column naming of processed dataframe to have target
    df['is_fraud'] = df[target_col].fillna(0).astype(int)
    
    return df, feature_cols, 'is_fraud', encoders

# ---------------------------------------------------------------------
# 3. Model Classes & Custom Autoencoder Wrapper
# ---------------------------------------------------------------------
class MLPAutoencoderWrapper:
    """Wrapper that mimics sklearn classifiers using MLPRegressor as an autoencoder."""
    def __init__(self, hidden_layer_sizes=(8, 4, 8), random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, 
            activation='relu', 
            solver='adam', 
            max_iter=100, 
            random_state=random_state
        )
        self.threshold = 0.0
        
    def fit(self, X, y=None):
        # Autoencoder trains only on normal data (is_fraud == 0)
        # If y is provided, filter X
        if y is not None:
            X_normal = X[y == 0]
        else:
            X_normal = X
            
        self.model.fit(X_normal, X_normal)
        
        # Calculate reconstruction error (MSE) on normal training data
        preds = self.model.predict(X_normal)
        mse = np.mean(np.power(X_normal - preds, 2), axis=1)
        # Set threshold at the 95th percentile of normal errors
        self.threshold = np.percentile(mse, 95)
        return self
        
    def predict(self, X):
        preds = self.model.predict(X)
        mse = np.mean(np.power(X - preds, 2), axis=1)
        return (mse > self.threshold).astype(int)
        
    def predict_proba(self, X):
        # Simulated probability based on reconstruction error relative to threshold
        preds = self.model.predict(X)
        mse = np.mean(np.power(X - preds, 2), axis=1)
        # Normalize so that values at the threshold are 0.5
        # Logistic sigmoid style
        scaled = mse / (self.threshold + 1e-9)
        probs = 1 / (1 + np.exp(-2.0 * (scaled - 1.0)))
        return np.column_stack((1 - probs, probs))

# ---------------------------------------------------------------------
# 4. Model Training and Loading
# ---------------------------------------------------------------------
def train_model(X_train, y_train, model_type='Random Forest', random_state=42):
    """Train a model of the specified type."""
    if model_type == 'Logistic Regression':
        clf = LogisticRegression(max_iter=500, random_state=random_state)
    elif model_type == 'Decision Tree':
        # Apply structural regularization
        clf = DecisionTreeClassifier(max_depth=8, min_samples_split=5, min_samples_leaf=3, random_state=random_state)
    elif model_type == 'Random Forest':
        # Apply structural regularization
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    elif model_type == 'XGBoost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, eval_metric='logloss', n_jobs=-1)
    elif model_type == 'LightGBM':
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state, verbosity=-1, n_jobs=-1)
    elif model_type == 'Isolation Forest':
        # Unsupervised model
        clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=random_state, n_jobs=-1)
        # Note: Isolation Forest outputs -1 (anomaly) and 1 (normal).
        # We wrapper it in evaluation
    elif model_type == 'One-Class SVM':
        clf = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    elif model_type == 'Autoencoder':
        clf = MLPAutoencoderWrapper(hidden_layer_sizes=(8, 4, 8), random_state=random_state)
    elif model_type == 'Ensemble Model':
        # Voting Classifier combining Logistic Regression, Random Forest, and XGBoost
        from xgboost import XGBClassifier
        lr = LogisticRegression(max_iter=500, random_state=random_state)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=random_state, eval_metric='logloss', n_jobs=-1)
        clf = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Fit the model
    if model_type in ['One-Class SVM']:
        # Trained only on normal data
        X_normal = X_train[y_train == 0]
        clf.fit(X_normal)
    elif model_type in ['Autoencoder']:
        clf.fit(X_train, y_train)
    elif model_type in ['Isolation Forest']:
        clf.fit(X_train)  # Unsupervised fit
    else:
        clf.fit(X_train, y_train)
        
    return clf

def evaluate_model_metrics(clf, X_test, y_test, model_type):
    """Evaluate a trained model and return a dictionary of metrics."""
    if model_type == 'Isolation Forest':
        # Outlier is -1, inlier is 1. Map to 1 (fraud) and 0 (legitimate)
        pred_raw = clf.predict(X_test)
        y_pred = np.where(pred_raw == -1, 1, 0)
        # Scores are negative of anomaly scores
        scores = -clf.score_samples(X_test)
        # scale scores to [0,1]
        probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    elif model_type == 'One-Class SVM':
        pred_raw = clf.predict(X_test)
        y_pred = np.where(pred_raw == -1, 1, 0)
        scores = clf.decision_function(X_test)
        # invert scores so that high value = fraud (outlier)
        scores = -scores
        probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    elif model_type == 'Autoencoder':
        y_pred = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
    else:
        y_pred = clf.predict(X_test)
        try:
            probs = clf.predict_proba(X_test)[:, 1]
        except AttributeError:
            probs = clf.decision_function(X_test)
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)
            
    # Calculate scores
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc_score = roc_auc_score(y_test, probs)
    except Exception:
        auc_score = 0.5
        
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc_score),
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }

def save_trained_model(model, filename):
    """Save model to models directory."""
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
        
def load_trained_model(filename):
    """Load model from models directory."""
    path = os.path.join(MODELS_DIR, filename)
    with open(path, 'rb') as f:
        return pickle.load(f)

# ---------------------------------------------------------------------
# 5. Graph-Based Analysis Functions
# ---------------------------------------------------------------------
def generate_transaction_graph(df, dataset_name, max_edges=150):
    """Build a NetworkX graph representing transactional relationships.
    Connects transactions sharing Locations, Merchant Categories, or User IDs.
    Returns: G (networkx graph), nodes_data, edges_data
    """
    G = nx.Graph()
    df_sample = df.head(max_edges)
    
    if dataset_name == 'synthetic':
        for idx, row in df_sample.iterrows():
            txn_id = f"TXN_{row['Transaction_ID']}"
            user_id = f"User_{row['User_ID']}"
            loc = f"Loc_{row['Location']}"
            merch = f"Merch_{row['Merchant_Category']}"
            
            # Nodes
            G.add_node(txn_id, type='transaction', is_fraud=int(row['is_fraud']), amount=row['Transaction_Amount'])
            G.add_node(user_id, type='user')
            G.add_node(loc, type='location')
            G.add_node(merch, type='merchant')
            
            # Edges
            G.add_edge(txn_id, user_id, type='belongs_to')
            G.add_edge(txn_id, loc, type='occurred_at')
            G.add_edge(txn_id, merch, type='purchased_from')
            
    elif dataset_name == 'credit_card':
        for idx, row in df_sample.iterrows():
            txn_id = f"TXN_{row['TransactionID']}"
            merch = f"Merch_{row['MerchantID']}"
            loc = f"Loc_{row['location']}"
            
            G.add_node(txn_id, type='transaction', is_fraud=int(row['is_fraud']), amount=row['amount'])
            G.add_node(merch, type='merchant')
            G.add_node(loc, type='location')
            
            G.add_edge(txn_id, merch, type='purchased_from')
            G.add_edge(txn_id, loc, type='occurred_at')
            
    else:  # financial
        for idx, row in df_sample.iterrows():
            txn_id = f"TXN_{row['Transaction_ID']}"
            cust_id = f"Cust_{row['Customer_ID']}"
            loc = f"Loc_{row['Location']}"
            merch = f"Merch_{row['Merchant_Category']}"
            
            G.add_node(txn_id, type='transaction', is_fraud=int(row['is_fraud']), amount=row['amount'])
            G.add_node(cust_id, type='user')
            G.add_node(loc, type='location')
            G.add_node(merch, type='merchant')
            
            G.add_edge(txn_id, cust_id, type='belongs_to')
            G.add_edge(txn_id, loc, type='occurred_at')
            G.add_edge(txn_id, merch, type='purchased_from')
            
    return G

# ---------------------------------------------------------------------
# 6. Real-Time Logging & Email Alert Pipeline
# ---------------------------------------------------------------------
ALERT_LOG_PATH = os.path.join(DATA_DIR, 'fraud_alerts.json')

def send_realtime_alert(txn_details, email_config=None):
    """Log an alert locally and optionally send an email alert."""
    alert_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    alert_entry = {
        'timestamp': alert_time,
        'transaction_details': txn_details,
        'status': 'logged'
    }
    
    # Save to local log
    alerts = []
    if os.path.exists(ALERT_LOG_PATH):
        try:
            with open(ALERT_LOG_PATH, 'r') as f:
                alerts = json.load(f)
        except Exception:
            alerts = []
            
    alerts.append(alert_entry)
    with open(ALERT_LOG_PATH, 'w') as f:
        json.dump(alerts, f, indent=4)
        
    # Email alert (if config is set and valid)
    if email_config and email_config.get('enabled'):
        try:
            sender_email = email_config.get('sender_email')
            sender_password = email_config.get('sender_password')
            recipient_email = email_config.get('recipient_email')
            
            if sender_email and sender_password and recipient_email:
                message = MIMEMultipart()
                message["From"] = sender_email
                message["To"] = recipient_email
                message["Subject"] = f"🚨 URGENT: Financial Fraud Alert ({txn_details.get('transaction_id', 'Unknown ID')})"
                
                body = f"""
                =======================================================
                🚨 SUSPICIOUS TRANSACTION DETECTED (FRAUD ALERT)
                =======================================================
                
                A high-risk transaction has been flagged by the Financial Fraud Detection System.
                
                Alert Timestamp: {alert_time}
                
                Transaction Highlights:
                -------------------------------------------------------
                - Transaction ID: {txn_details.get('transaction_id', 'N/A')}
                - Amount: ${txn_details.get('amount', 'N/A'):,.2f}
                - Location: {txn_details.get('location', 'N/A')}
                - Merchant Category: {txn_details.get('merchant_category', 'N/A')}
                - Fraud Probability: {txn_details.get('fraud_probability', 0.0):.2f}%
                
                Action Required:
                Please review this transaction and temporarily freeze the card if necessary.
                
                =======================================================
                This is an automated alert generated by your custom ML pipeline.
                """
                message.attach(MIMEText(body, "plain"))
                
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, message.as_string())
                server.quit()
                
                alert_entry['status'] = 'emailed'
                # rewrite
                with open(ALERT_LOG_PATH, 'w') as f:
                    json.dump(alerts, f, indent=4)
                return True, "Alert logged and email sent successfully!"
        except Exception as e:
            return False, f"Alert logged but email failed: {str(e)}"
            
    return True, "Alert logged successfully to local records (email not configured)."