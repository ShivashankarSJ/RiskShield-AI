import sys, os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# Add the project root to the Python path so utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.helpers import load_data, preprocess_data, train_model, load_model

# Global path to the serialized model file
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'fraud_detection_model.pkl'))


st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Sidebar navigation 
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Explorer", "Model Performance", "Real‑Time Prediction"],
    index=0,
)

@st.cache_data
def get_data():
    """Load raw CSV and return a cleaned DataFrame."""
    df_raw = load_data()            # utils/helpers.load_data loads the CSV
    df_clean = preprocess_data(df_raw)  # standardize columns, extract features
    return df_raw, df_clean

raw_df, df = get_data()

# Helper 
def show_key_metrics(df_original, df_clean, model_accuracy=None):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df_original):,}")
    # Detect fraud column name (original CSV uses "IsFraud")
    fraud_col = "IsFraud" if "IsFraud" in df_original.columns else "is_fraud"
    fraud_cnt = df_original[fraud_col].sum()
    col2.metric("Fraudulent Txns", f"{fraud_cnt:,}")
    if model_accuracy is not None:
        col3.metric("Model Accuracy", f"{model_accuracy:.2f}%")
    else:
        col3.metric("Model Accuracy", "—")

# SECTION: Overview 
if section == "Overview":
    st.title("🕵️‍♀️ Financial Fraud Detection Dashboard")
    st.markdown(
        """
        This dashboard visualises a **machine‑learning model** that flags potentially fraudulent
        financial transactions. It provides:

        - **Real‑time prediction** – enter a transaction and instantly see the fraud risk.
        - **Model performance** – accuracy, confusion matrix, and feature importance.
        - **Exploratory data analysis** – interactive charts to spot patterns.
        """
    )
    show_key_metrics(raw_df, df)
    st.success("Data loaded successfully – navigate using the sidebar.")

#  Data Explorer
elif section == "Data Explorer":
    st.header("📊 Exploratory Data Analysis")
    # Transaction amount distribution (log scale)
    fig_amount = px.histogram(
        df,
        x="amount",
        nbins=50,
        title="Transaction Amount Distribution",
        log_x=True,
        color_discrete_sequence=["#636EFA"],
    )
    st.plotly_chart(fig_amount, use_container_width=True)

    # Fraud rate by hour of day (if hour column exists after preprocessing)
    if "hour" in df.columns:
        hourly = df.groupby("hour")["is_fraud"].mean().reset_index()
        fig_hour = px.line(
            hourly,
            x="hour",
            y="is_fraud",
            title="Fraud Rate by Hour of Day",
            markers=True,
            color_discrete_sequence=["#EF553B"],
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    # Optional geographic map – show if latitude/longitude columns are present
    geo_cols = [c for c in ["latitude", "longitude", "lat", "lng"] if c in df.columns]
    if len(geo_cols) == 2:
        st.subheader("🌍 Transaction Locations")
        st.map(df[[geo_cols[0], geo_cols[1]]].dropna())

    # Expandable raw data view (first 200 rows)
    with st.expander("Show raw data (first 200 rows)"):
        st.dataframe(raw_df.head(200))

# Model Performance 
elif section == "Model Performance":
    st.header("📈 Model Performance")
    model_path = "..\\models\\fraud_detection_model.pkl"
    # Load pre‑trained model if it exists, otherwise train a new one
    try:
        model = load_model(model_path)
        st.success("Loaded pre‑trained model.")
    except Exception:
        st.info("Training a new model – this may take a few seconds…")
        model = train_model(df)
        # Save for future runs
        import os, pickle
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        st.success("Model trained and saved.")

    # Split data for evaluation (mirrors utils/helpers.train_model)
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[col for col in ["is_fraud", "IsFraud"] if col in df.columns])
    y = df["is_fraud"] if "is_fraud" in df.columns else df["IsFraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test) * 100

    # Show metrics at top of page
    show_key_metrics(raw_df, df, model_accuracy=accuracy)

    # Confusion matrix – Plotly heatmap
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Classification report – plain text output
    report = classification_report(y_test, y_pred)
    st.text(report)

    # Feature importance – bar chart if estimator provides it
    if hasattr(model.named_steps["classifier"], "feature_importances_"):
        importances = model.named_steps["classifier"].feature_importances_
        features = X.columns
        imp_df = pd.DataFrame({"feature": features, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        fig_imp = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance",
            color="importance",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Model does not expose feature importances.")

# Real‑Time Prediction 
else:  # Real‑Time Prediction
    st.header("🔎 Real‑Time Fraud Prediction")
    # Ensure model is loaded (training block above guarantees it)
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model: {e}")
        st.stop()

    # Input form in the sidebar
    st.sidebar.subheader("Enter transaction details")
    amount = st.sidebar.number_input("Amount ($)", min_value=0.0, value=500.0, step=10.0)
    hour = st.sidebar.slider("Hour of day", 0, 23, 12)
    day_of_week = st.sidebar.selectbox("Day of week", list(range(7)), index=0)
    transaction_type = st.sidebar.selectbox(
        "Transaction type",
        ["purchase", "refund", "withdrawal", "transfer"],
        index=0,
    )

    # Build a DataFrame that matches the training feature set
    input_df = pd.DataFrame({
        "amount": [amount],
        "hour": [hour],
        "day_of_week": [day_of_week],
        "amount_log": [np.log1p(amount)],
        "transaction_type": [transaction_type],
    })

    if st.sidebar.button("Predict Fraud Risk"):
        try:
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1] * 100
            if pred == 1:
                st.error(f"🚨 **Fraud Detected!** Probability: {prob:.2f}%")
            else:
                st.success(f"✅ **Legitimate** – Probability of fraud: {prob:.2f}%")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Footer 
st.markdown("---")
st.caption("© 2026 Financial Fraud Detection Project – built with Streamlit, scikit‑learn, and Plotly")
