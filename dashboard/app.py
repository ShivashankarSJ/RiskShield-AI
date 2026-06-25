import sys
import os
import sqlite3
import json
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import (
    preprocess_dataset, DB_PATH, MODELS_DIR, ALERT_LOG_PATH, 
    generate_transaction_graph, send_realtime_alert, load_trained_model
)

# ---------------------------------------------------------------------
# UI Config & Design Details
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="RiskShield AI | Enterprise Fraud Detection Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    /* Styling headers */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #852020);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    .kpi-container {
        background-color: #1E1E24;
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 5px solid #FF4B4B;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .kpi-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #AAAAAA;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
@st.cache_data
def get_db_data(table_name):
    """Retrieve raw rows from SQLite."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def load_metrics():
    """Load model metrics saved in JSON."""
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}

def load_feature_names():
    """Load trained feature names from JSON."""
    feat_path = os.path.join(MODELS_DIR, 'feature_names.json')
    if os.path.exists(feat_path):
        with open(feat_path, 'r') as f:
            return json.load(f)
    return []

# ---------------------------------------------------------------------
# Sidebar & State Management
# ---------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/nolan/96/shield.png", width=90)
st.sidebar.markdown("<h2 style='margin-top:0px;'>RiskShield AI</h2>", unsafe_allow_html=True)
st.sidebar.caption("Financial Fraud Detection System v2.0")

# Dataset Selector
dataset_choice = st.sidebar.selectbox(
    "Select Transaction Database",
    ["Synthetic Fraud (50k rows)", "Credit Card Fraud (100k rows)", "Financial Transactions (5k rows)"],
    index=0
)

# Map human choice to SQLite table names
table_mapping = {
    "Synthetic Fraud (50k rows)": "synthetic_transactions",
    "Credit Card Fraud (100k rows)": "credit_card_transactions",
    "Financial Transactions (5k rows)": "financial_transactions"
}
dataset_mapping = {
    "Synthetic Fraud (50k rows)": "synthetic",
    "Credit Card Fraud (100k rows)": "credit_card",
    "Financial Transactions (5k rows)": "financial"
}

table_name = table_mapping[dataset_choice]
dataset_name = dataset_mapping[dataset_choice]

# Load Data
df_raw = get_db_data(table_name)

if df_raw.empty:
    st.error(f"Failed to load database table '{table_name}'. Please verify that the ETL pipeline has run.")
    st.stop()

# Preprocess for statistics & visualizations
df_clean, features, target_col, encoders = preprocess_dataset(df_raw, dataset_name)

# Model Metrics
metrics_dict = load_metrics()

# Navigation
section = st.sidebar.radio(
    "Modules",
    ["Executive Overview", "Exploratory Analytics", "Graph Anomaly Network", "Model Arena", "Streaming Simulator & Form"],
    index=0
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("Database Connected")
st.sidebar.info(f"Loaded Table: `{table_name}` ({len(df_raw):,} records)")

# ---------------------------------------------------------------------
# MODULE 1: Executive Overview
# ---------------------------------------------------------------------
if section == "Executive Overview":
    st.markdown("<h1 class='main-title'>🛡️ RiskShield AI Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Real-time financial anomaly monitoring and predictive fraud modeling platform</p>", unsafe_allow_html=True)
    
    # KPIs Row
    total_txns = len(df_raw)
    fraud_col_name = 'Fraud_Label' if 'Fraud_Label' in df_raw.columns else ('IsFraud' if 'IsFraud' in df_raw.columns else 'Fraudulent')
    fraud_txns = int(df_raw[fraud_col_name].fillna(0).sum())
    fraud_rate = (fraud_txns / total_txns) * 100 if total_txns > 0 else 0.0
    
    # Detect Amount Column
    amt_col = 'Transaction_Amount' if 'Transaction_Amount' in df_raw.columns else ('Amount' if 'Amount' in df_raw.columns else 'Transaction_Amount')
    avg_amt = float(df_raw[amt_col].fillna(0).mean()) if amt_col in df_raw.columns else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-label'>Total Transactions</div>
            <div class='kpi-number'>{total_txns:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='kpi-container' style='border-left-color: #E74C3C;'>
            <div class='kpi-label'>Fraudulent Activities</div>
            <div class='kpi-number' style='color: #E74C3C;'>{fraud_txns:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='kpi-container' style='border-left-color: #E67E22;'>
            <div class='kpi-label'>Fraud Rate</div>
            <div class='kpi-number' style='color: #E67E22;'>{fraud_rate:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='kpi-container' style='border-left-color: #2ECC71;'>
            <div class='kpi-label'>Average Volume</div>
            <div class='kpi-number' style='color: #2ECC71;'>${avg_amt:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### System Summary")
    st.write(
        "RiskShield AI connects directly to banking records via an automated SQLite ETL pipeline. "
        "It applies advanced supervised and unsupervised machine learning algorithms to detect transaction patterns "
        "associated with credit card cloning, wire fraud, and coordinated network ring fraud."
    )
    
    # Sample Plotly visualization
    st.markdown("### Recent Transactions Analysis")
    fig_hist = px.histogram(
        df_clean,
        x="amount_log" if "amount_log" in df_clean.columns else "amount",
        color="is_fraud",
        nbins=60,
        title="Transaction Volume Distribution (Log-Scale) by Fraud Label",
        color_discrete_map={0: "#2ECC71", 1: "#E74C3C"},
        labels={"is_fraud": "Is Fraudulent?", "amount_log": "Log of Transaction Amount", "amount": "Transaction Amount"},
        barmode="overlay",
        height=450
    )
    fig_hist.update_layout(template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------
# MODULE 2: Exploratory Analytics
# ---------------------------------------------------------------------
elif section == "Exploratory Analytics":
    st.markdown("<h1 class='main-title'>📊 Exploratory Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Interactive data visualization and real-time database querying</p>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Visual Analytics", "Outlier Analysis", "Database ETL Console"])
    
    with tab1:
        st.subheader("Statistical Distributions")
        
        # Categorical Distribution
        cat_cols = [c for c in ['Transaction_Type', 'Device_Type', 'Payment_Method', 'Merchant_Category'] if c in df_clean.columns]
        if cat_cols:
            selected_cat = st.selectbox("Select Categorical Variable for Analysis", cat_cols)
            
            # Bar plot of fraud rates
            grouped = df_clean.groupby(selected_cat)['is_fraud'].agg(['count', 'mean']).reset_index()
            grouped['fraud_rate_%'] = grouped['mean'] * 100
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig_count = px.bar(
                    grouped, x=selected_cat, y='count', 
                    title=f"Total Transactions by {selected_cat}",
                    color='count', color_continuous_scale="Viridis",
                    height=400
                )
                fig_count.update_layout(template="plotly_dark")
                st.plotly_chart(fig_count, use_container_width=True)
                
            with col_b:
                fig_rate = px.bar(
                    grouped, x=selected_cat, y='fraud_rate_%', 
                    title=f"Fraud Rate (%) by {selected_cat}",
                    color='fraud_rate_%', color_continuous_scale="Reds",
                    height=400
                )
                fig_rate.update_layout(template="plotly_dark")
                st.plotly_chart(fig_rate, use_container_width=True)
        else:
            st.info("No categorical columns available for this dataset.")
            
        # Cyclical behaviour
        st.subheader("Temporal Trends")
        if 'hour' in df_clean.columns:
            hourly_fraud = df_clean.groupby('hour')['is_fraud'].mean().reset_index()
            hourly_fraud['fraud_rate_%'] = hourly_fraud['is_fraud'] * 100
            
            fig_hour = px.line(
                hourly_fraud, x='hour', y='fraud_rate_%',
                title="Fraud Rate (%) by Hour of Day",
                markers=True,
                color_discrete_sequence=["#FF4B4B"],
                height=350
            )
            fig_hour.update_layout(template="plotly_dark")
            st.plotly_chart(fig_hour, use_container_width=True)
        else:
            st.info("No hourly temporal data found in this dataset.")
            
    with tab2:
        st.subheader("Outlier & Feature Correlation")
        
        # Spearman correlation heatmap of numerical cols
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ['is_fraud', 'transaction_id', 'user_id', 'customer_id', 'merchant_id']]
        
        if num_cols:
            col_x, col_y = st.columns([1, 2])
            with col_x:
                st.markdown("**Boxplots for Outlier Detection**")
                box_col = st.selectbox("Select Numeric Feature for Boxplot", num_cols)
                fig_box = px.box(
                    df_clean, y=box_col, color="is_fraud",
                    title=f"Distribution of {box_col} for Outliers",
                    color_discrete_map={0: "#2ECC71", 1: "#E74C3C"},
                    height=380
                )
                fig_box.update_layout(template="plotly_dark")
                st.plotly_chart(fig_box, use_container_width=True)
                
            with col_y:
                st.markdown("**Spearman Feature Correlation**")
                corr = df_clean[num_cols].corr(method='spearman')
                fig_heat = px.imshow(
                    corr, text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Matrix Heatmap",
                    height=380
                )
                fig_heat.update_layout(template="plotly_dark")
                st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No numeric features to perform outlier correlation.")
            
    with tab3:
        st.subheader("Direct SQLite Access Console")
        st.write(
            "This tab simulates an ETL environment. You can write custom SQL queries against the local "
            "SQLite database table containing the raw records."
        )
        
        # Preset queries
        preset = st.selectbox(
            "Choose a Preset Query",
            [
                "Custom SQL Query",
                f"SELECT * FROM {table_name} WHERE is_fraud = 1 LIMIT 5" if 'is_fraud' in df_clean.columns else f"SELECT * FROM {table_name} LIMIT 5",
                f"SELECT Location, count(*) as count FROM {table_name} GROUP BY Location ORDER BY count DESC LIMIT 10",
                f"SELECT Transaction_Type, avg(Transaction_Amount) as avg_amt FROM {table_name} GROUP BY Transaction_Type" if dataset_name == 'synthetic' else f"SELECT * FROM {table_name} LIMIT 5"
            ]
        )
        
        if preset == "Custom SQL Query":
            default_q = f"SELECT * FROM {table_name} LIMIT 10"
        else:
            default_q = preset
            
        sql_input = st.text_area("Write SQL Query", default_q, height=100)
        
        if st.button("Execute SQL"):
            try:
                conn = sqlite3.connect(DB_PATH)
                res_df = pd.read_sql_query(sql_input, conn)
                conn.close()
                st.success("Query executed successfully!")
                st.dataframe(res_df)
            except Exception as e:
                st.error(f"SQL Error: {e}")

# ---------------------------------------------------------------------
# MODULE 3: Graph Anomaly Network
# ---------------------------------------------------------------------
elif section == "Graph Anomaly Network":
    st.markdown("<h1 class='main-title'>🕸️ Graph Anomaly Network</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Analyzing fraudulent network rings and multi-hop transactional paths</p>", unsafe_allow_html=True)
    
    st.write(
        "By mapping transactions to graph nodes sharing entity relationships (e.g. same user device, "
        "merchant category, or geographic location), we can identify **fraud networks** and money laundering rings."
    )
    
    limit_edges = st.slider("Select transaction sample size (edges) for graph representation", 30, 200, 80, step=10)
    
    with st.spinner("Generating network graph structure..."):
        G = generate_transaction_graph(df_clean, dataset_name, max_edges=limit_edges)
        
        # Graph metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
        
        cols_metric = st.columns(4)
        cols_metric[0].metric("Graph Nodes", num_nodes)
        cols_metric[1].metric("Graph Edges", num_edges)
        cols_metric[2].metric("Network Density", f"{density:.4f}")
        cols_metric[3].metric("Avg Node Connection", f"{avg_degree:.2f}")
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=0.15, seed=42)
        
        # Edges layout
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#555555'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Nodes layout
        node_x = []
        node_y = []
        node_color = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Attributes
            node_type = G.nodes[node].get('type', 'entity')
            is_fraud = G.nodes[node].get('is_fraud', 0)
            
            if node_type == 'transaction':
                node_text.append(f"Txn: {node}<br>Fraud: {is_fraud}<br>Amount: ${G.nodes[node].get('amount', 0.0):.2f}")
                node_color.append('#E74C3C' if is_fraud == 1 else '#2ECC71')
                node_size.append(15)
            else:
                node_text.append(f"Entity: {node}<br>Type: {node_type}")
                node_color.append('#2980B9')
                node_size.append(9)
                
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_color,
                size=node_size,
                line_width=1.5
            )
        )
        
        # Draw Plotly Graph
        fig_graph = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text='<br>Transactional Link Network (Red = Flagged Fraud, Green = Legitimate, Blue = Shared Entities)',
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )
        fig_graph.update_layout(template="plotly_dark")
        st.plotly_chart(fig_graph, use_container_width=True)

# ---------------------------------------------------------------------
# MODULE 4: Model Arena
# ---------------------------------------------------------------------
elif section == "Model Arena":
    st.markdown("<h1 class='main-title'>📈 Machine Learning Model Arena</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Comparing supervised, unsupervised, and deep anomaly detection models</p>", unsafe_allow_html=True)
    
    if not metrics_dict:
        st.warning("Performance metrics file not found. Please verify that `predict_fraud.py` trained all models and saved metrics.")
        st.stop()
        
    # Standard Table comparison
    st.subheader("General Metrics Summary")
    
    records = []
    for model_name, metrics in metrics_dict.items():
        records.append({
            "Model Name": model_name,
            "Accuracy": f"{metrics['accuracy']*100:.2f}%",
            "Precision": f"{metrics['precision']*100:.2f}%",
            "Recall": f"{metrics['recall']*100:.2f}%",
            "F1-Score": f"{metrics['f1_score']*100:.2f}%",
            "ROC AUC": f"{metrics['roc_auc']:.4f}"
        })
    df_metrics = pd.DataFrame(records)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # ------------------------------------------------------------
    # ROC Curves side-by-side
    # ------------------------------------------------------------
    st.subheader("Model Evaluation Charts")
    col_c1, col_c2 = st.columns([3, 2])
    
    with col_c1:
        # Plot ROC curves of all models
        fig_roc = go.Figure()
        for model_name, metrics in metrics_dict.items():
            if 'fpr_curve' in metrics and 'tpr_curve' in metrics:
                fig_roc.add_trace(go.Scatter(
                    x=metrics['fpr_curve'], 
                    y=metrics['tpr_curve'],
                    mode='lines',
                    name=f"{model_name} (AUC={metrics['roc_auc']:.2f})"
                ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random Guess'))
        fig_roc.update_layout(
            title="ROC Curves Comparison (Receiver Operating Characteristic)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=450,
            template="plotly_dark"
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
    with col_c2:
        # Detailed single-model view
        selected_model = st.selectbox("Select Model for Detailed Examination", list(metrics_dict.keys()))
        metrics_sm = metrics_dict[selected_model]
        
        # Confusion matrix
        cm = np.array(metrics_sm['confusion_matrix'])
        cm_df = pd.DataFrame(cm, index=["Legit (0)", "Fraud (1)"], columns=["Pred Legit (0)", "Pred Fraud (1)"])
        fig_cm = px.imshow(
            cm_df, text_auto=True,
            color_continuous_scale="Reds",
            title=f"{selected_model} Confusion Matrix",
            height=300
        )
        fig_cm.update_layout(template="plotly_dark")
        st.plotly_chart(fig_cm, use_container_width=True)
        
    # Feature Importance (if RF or Decision Tree selected)
    st.markdown("### Model Architecture Details")
    desc_cols = st.columns(2)
    with desc_cols[0]:
        st.markdown(f"**Structural Details: {selected_model}**")
        if selected_model == 'Autoencoder':
            st.write(
                "Our **Autoencoder** is trained strictly on legitimate transaction records (unsupervised anomaly detection). "
                "The neural network learns a bottleneck reconstruction representation (hidden layers of `8 -> 4 -> 8` neurons). "
                "During scoring, suspicious transactions show a significantly higher Reconstruction Mean Squared Error (MSE), "
                "which flags them as fraudulent anomalies. This handles novel fraud patterns unseen in labels."
            )
        elif selected_model == 'Isolation Forest':
            st.write(
                "**Isolation Forest** isolates anomalies by randomly partitioning feature boundaries. "
                "Legitimate nodes require many splits to isolate, while fraudulent ones sit at very short paths. "
                "This tree-based unsupervised algorithm is ideal for high-dimensional financial transaction data."
            )
        elif selected_model == 'Ensemble Model':
            st.write(
                "Our **Voting Classifier Ensemble** aggregates predictions from Logistic Regression, "
                "Random Forest, and XGBoost using soft voting. Combining multiple base classifiers "
                "reduces individual variance and strengthens overall recall on complex transactions."
            )
        elif selected_model in ['Decision Tree', 'Random Forest']:
            st.write(
                "To prevent overfitting on imbalanced banking data, we applied structural regularization. "
                "This includes setting a strict `max_depth` restriction, requiring `min_samples_split` values, "
                "and setting leaf thresholds. This guarantees stable, generalizable decision rules."
            )
        else:
            st.write(
                "Standard classification model mapping numeric features to transaction probabilities. "
                "Preprocessed with standard MinMaxScaler normalization for uniform distance weights."
            )
            
    with desc_cols[1]:
        # Feature Importance representation if Random Forest exists
        try:
            model_filename = selected_model.lower().replace(' ', '_').replace('-', '_') + '.pkl'
            model_obj = load_trained_model(model_filename)
            
            # Check classifier importance
            clf_step = model_obj
            if hasattr(clf_step, 'named_steps'): # In case of sklearn pipeline (if any)
                clf_step = clf_step.named_steps.get('classifier', clf_step)
                
            if hasattr(clf_step, 'feature_importances_'):
                importances = clf_step.feature_importances_
                feat_names = load_feature_names()
                
                if len(importances) == len(feat_names):
                    imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
                    imp_df = imp_df.sort_values('Importance', ascending=True)
                    
                    fig_feat = px.bar(
                        imp_df, x='Importance', y='Feature', orientation='h',
                        title=f"{selected_model} Feature Importance",
                        color='Importance', color_continuous_scale="Bluered",
                        height=280
                    )
                    fig_feat.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info("Feature count mismatch or feature importances not matching names.")
            else:
                st.info("This model type does not support direct feature importance inspection.")
        except Exception:
            st.info("Feature importance inspection unavailable for the selected model.")

# ---------------------------------------------------------------------
# MODULE 5: Streaming Simulator & Prediction Form
# ---------------------------------------------------------------------
elif section == "Streaming Simulator & Form":
    st.markdown("<h1 class='main-title'>⚡ Streaming Simulator & Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Simulating live bank streaming logs and testing individual transactions</p>", unsafe_allow_html=True)
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        st.subheader("Interactive Transaction Scoring Form")
        st.write("Enter transaction features to run an instantaneous prediction request through a pre-trained model.")
        
        # Load models list
        trained_models = [m for m in list(metrics_dict.keys()) if m not in ['Isolation Forest', 'One-Class SVM']]
        if not trained_models:
            trained_models = ["Random Forest"]
            
        predict_model_choice = st.selectbox("Active Scoring Model", trained_models)
        
        # Custom parameters depending on selected dataset features
        st.markdown("**Transaction Details**")
        amount_input = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=250.0)
        acc_bal_input = st.number_input("Account Balance ($)", min_value=0.0, max_value=1000000.0, value=12400.0)
        daily_count = st.slider("Daily Transaction Count", 1, 50, 4)
        card_age = st.slider("Card Age (Days)", 1, 365, 120)
        prev_fraud = st.selectbox("Previous Fraudulent Activity?", [0, 1], index=0)
        
        # Categorical choices
        txn_type_choice = st.selectbox("Transaction Type", ["POS", "Online", "ATM Withdrawal", "Bank Transfer"])
        device_choice = st.selectbox("Device Type", ["Mobile", "Tablet", "Laptop", "Desktop"])
        location_choice = st.selectbox("Location Country/City", ["Mumbai", "New York", "London", "Tokyo", "Sydney"])
        merchant_choice = st.selectbox("Merchant Category", ["Travel", "Groceries", "Restaurants", "Electronics", "Clothing"])
        card_type_choice = st.selectbox("Card Type", ["Visa", "Mastercard", "Amex", "Discover"])
        
        if st.button("Perform Instant Score"):
            try:
                # Load models
                model_filename = predict_model_choice.lower().replace(' ', '_').replace('-', '_') + '.pkl'
                model_obj = load_trained_model(model_filename)
                
                # Load scaler & encoders
                with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
                    scaler_obj = pickle.load(f)
                with open(os.path.join(MODELS_DIR, 'encoders.pkl'), 'rb') as f:
                    encoders_obj = pickle.load(f)
                    
                # Setup categorical values
                # If encoder doesn't know value, fit dummy or grab first index
                def encode_cat(col_name, val):
                    le = encoders_obj.get(col_name)
                    if le:
                        try:
                            return le.transform([val])[0]
                        except Exception:
                            return 0
                    return 0
                    
                amount_log = np.log1p(amount_input)
                day_of_week = 3 # mock
                
                # Align with features list
                feat_names = load_feature_names()
                
                input_data = {
                    'Transaction_Amount': amount_input,
                    'Account_Balance': acc_bal_input,
                    'Previous_Fraudulent_Activity': prev_fraud,
                    'Daily_Transaction_Count': daily_count,
                    'Card_Age': card_age,
                    'amount_log': amount_log,
                    'day_of_week': day_of_week,
                    'transaction_type_encoded': encode_cat('Transaction_Type', txn_type_choice),
                    'device_type_encoded': encode_cat('Device_Type', device_choice),
                    'location_encoded': encode_cat('Location', location_choice),
                    'merchant_category_encoded': encode_cat('Merchant_Category', merchant_choice),
                    'card_type_encoded': encode_cat('Card_Type', card_type_choice)
                }
                
                # Construct feature vector
                row_vec = []
                for fn in feat_names:
                    row_vec.append(input_data.get(fn, 0.0))
                    
                row_df = pd.DataFrame([row_vec], columns=feat_names)
                
                # Standard scaling
                scaled_df = pd.DataFrame(scaler_obj.transform(row_df), columns=feat_names)
                
                # Predict
                prediction = model_obj.predict(scaled_df)[0]
                prob = model_obj.predict_proba(scaled_df)[0][1] * 100
                
                st.markdown("---")
                if prediction == 1:
                    st.error(f"🚨 **SUSPICIOUS TRANSACTION DETECTED** (Probability: {prob:.2f}%)")
                    st.caption("This transaction matches known fraudulent sequences. Action: Lock Card.")
                else:
                    st.success(f"✅ **TRANSACTION APPROVED** (Probability of fraud: {prob:.2f}%)")
                    st.caption("Transaction normal. Action: Approve.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                
    with col_sim2:
        st.subheader("Live Transaction Streaming Simulator")
        st.write(
            "Simulates a stream of live bank logs pulled from SQLite. "
            "Flagged fraud triggers visual alerts and writes records to the local alert log file."
        )
        
        # Stream Settings
        model_choice_stream = st.selectbox("Streaming Detection Model", list(metrics_dict.keys()), key="stream_model")
        sim_speed = st.slider("Simulation Speed (seconds per transaction)", 0.2, 3.0, 1.0)
        
        # Email settings toggle
        st.markdown("**Email Alert Pipeline Settings**")
        email_enabled = st.checkbox("Enable SMTP Email Alerts for Fraud Flags")
        
        email_config = {"enabled": email_enabled}
        if email_enabled:
            email_config["sender_email"] = st.text_input("Sender Gmail Address", "my_alerts_bot@gmail.com")
            email_config["sender_password"] = st.text_input("App-Specific Gmail Password", type="password")
            email_config["recipient_email"] = st.text_input("Recipient Email Address", "analyst@bank.com")
            st.caption("Note: Gmail requires setting up an App Password (SMTP TLS, 587) for security.")
            
        start_btn = st.button("Start Live Simulation Stream")
        
        if start_btn:
            try:
                # Load models
                model_filename = model_choice_stream.lower().replace(' ', '_').replace('-', '_') + '.pkl'
                model_obj = load_trained_model(model_filename)
                
                # Load scaler
                with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'rb') as f:
                    scaler_obj = pickle.load(f)
                    
                feat_names = load_feature_names()
                
                # Sample random rows from SQLite that might have fraud
                conn = sqlite3.connect(DB_PATH)
                sim_rows = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 20", conn)
                conn.close()
                
                # Process simulator rows
                sim_rows_proc, _, _, _ = preprocess_dataset(sim_rows, dataset_name)
                
                st.markdown("---")
                st.info(f"Streaming Active... Scanning using model `{model_choice_stream}`")
                
                log_placeholder = st.empty()
                alert_placeholder = st.empty()
                
                log_text = ""
                for idx, row in sim_rows_proc.iterrows():
                    # Align with features list
                    row_vec = []
                    for fn in feat_names:
                        row_vec.append(row.get(fn, 0.0))
                        
                    row_df = pd.DataFrame([row_vec], columns=feat_names)
                    scaled_df = pd.DataFrame(scaler_obj.transform(row_df), columns=feat_names)
                    
                    # Score
                    if model_choice_stream == 'Isolation Forest':
                        pred_raw = model_obj.predict(scaled_df)[0]
                        prediction = 1 if pred_raw == -1 else 0
                        prob = float(-model_obj.score_samples(scaled_df)[0] * 100)
                    elif model_choice_stream == 'One-Class SVM':
                        pred_raw = model_obj.predict(scaled_df)[0]
                        prediction = 1 if pred_raw == -1 else 0
                        prob = float(-model_obj.decision_function(scaled_df)[0] * 100)
                    elif model_choice_stream == 'Autoencoder':
                        prediction = model_obj.predict(scaled_df)[0]
                        prob = float(model_obj.predict_proba(scaled_df)[0][1] * 100)
                    else:
                        prediction = model_obj.predict(scaled_df)[0]
                        prob = float(model_obj.predict_proba(scaled_df)[0][1] * 100)
                        
                    # Build details for logging
                    txn_id = row.get('Transaction_ID', row.get('TransactionID', f"TXN_{idx}"))
                    amt_val = float(row.get('Transaction_Amount', row.get('amount', 0.0)))
                    loc_val = row.get('Location', row.get('location', 'Unknown'))
                    merch_val = row.get('Merchant_Category', row.get('merchant_id', 'Unknown'))
                    
                    details = {
                        'transaction_id': str(txn_id),
                        'amount': amt_val,
                        'location': str(loc_val),
                        'merchant_category': str(merch_val),
                        'fraud_probability': prob
                    }
                    
                    if prediction == 1:
                        log_line = f"🚨 ALERT! [{time.strftime('%H:%M:%S')}] Suspicious Txn {txn_id} | Amount: ${amt_val:.2f} | Risk: {prob:.1f}%\n"
                        log_text = log_line + log_text
                        
                        alert_placeholder.error(f"🚨 **FRAUD DETECTED**: Txn {txn_id} of ${amt_val:.2f} at {loc_val}. System sent notification.")
                        
                        # Trigger Alert Function
                        send_realtime_alert(details, email_config)
                    else:
                        log_line = f"✅ OK [{time.strftime('%H:%M:%S')}] Txn {txn_id} | Amount: ${amt_val:.2f} | Risk: {prob:.1f}%\n"
                        log_text = log_line + log_text
                        
                    log_placeholder.code(log_text)
                    time.sleep(sim_speed)
                    
                st.success("Simulation batch complete.")
                
            except Exception as e:
                st.error(f"Simulator Error: {e}")

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.caption("🛡️ RiskShield AI Enterprise Fraud Suite | Built by shivashankar S J for Data Science & Analytics Internships")