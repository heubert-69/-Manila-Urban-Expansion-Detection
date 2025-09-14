import os
import pickle
import warnings
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap

warnings.filterwarnings("ignore")

# -----------------------------
# Load model
# -----------------------------
def load_model(model_name="model.pkl"):
    possible_paths = [
        model_name,
        os.path.join(os.getcwd(), model_name),
        f"/app/{model_name}",
        f"./{model_name}",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)

                # Case 1: dict (your earlier version)
                if isinstance(obj, dict):
                    st.sidebar.success(f"✅ Loaded dict model from: {path}")
                    return obj

                # Case 2: Pipeline/model only
                else:
                    st.sidebar.success(f"✅ Loaded sklearn pipeline/model from: {path}")
                    return {"model": obj, "scaler": None, "feature_names": []}

            except Exception as e:
                st.sidebar.error(f"❌ Error loading model from {path}: {e}")
                return None
    st.sidebar.error("❌ No model.pkl found in expected paths.")
    return None


model_data = load_model("model.pkl")
# -----------------------------
# Prediction
# -----------------------------
def predict_urbanization_csv(csv_file):
    if model_data is None:
        st.warning("⚠️ No model found. Running in demo mode.")
        return None, None, None, None

    model = model_data["model"]
    scaler = model_data.get("scaler", None)
    expected_features = model_data.get("feature_names", [])

    df = pd.read_csv(csv_file)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())
    df = df.clip(lower=-1e6, upper=1e6)
    st.write(f"📁 CSV loaded with shape: {df.shape}")

    # If model_data doesn’t store feature names → infer from model if possible
    if not expected_features and hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    elif not expected_features:
        expected_features = df.columns.tolist()

    # Check missing features
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        st.error(f"❌ Missing required features: {', '.join(missing_features)}")
        return None, None, None, None

    # Drop extra columns not in expected features
    X = df[expected_features].copy()

    # Handle NaNs
    if X.isnull().any().any():
        X = X.fillna(X.median())
        st.warning("⚠️ Missing values filled with median.")

    # Scale or let pipeline handle it
    if scaler:
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
    else:
        X_scaled = X.values
        predictions = model.predict(X)

    # Probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = predictions.astype(float)

    results_df = df.copy()
    results_df["prediction"] = predictions
    results_df["prediction_label"] = results_df["prediction"].map({0: "Non-Urban", 1: "Urban"})
    results_df["probability_urban"] = probabilities
    results_df["confidence"] = np.abs(probabilities - 0.5) * 2

    return results_df, probabilities, X_scaled, expected_features


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛰️ Manila Urban Expansion Prediction Tool")
st.markdown("""
Upload a **CSV with spectral features** (from Landsat or other sources)  
to predict **urban vs non-urban areas** using a pre-trained ML model.
""")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df, probs, X_scaled, feature_names = predict_urbanization_csv(uploaded_file)

    if df is not None:
        # Summary
        urban_count = (df["prediction"] == 1).sum()
        total_count = len(df)
        urban_pct = urban_count / total_count * 100

        st.subheader("📊 Results Summary")
        st.write(f"**Total samples**: {total_count}")
        st.write(f"**Urban areas predicted**: {urban_count} ({urban_pct:.1f}%)")

        # Distribution plots
        fig, ax = plt.subplots()
        ax.bar(["Non-Urban", "Urban"], [total_count - urban_count, urban_count], color=["green", "red"])
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.hist(probs, bins=20, color="blue", edgecolor="black")
        ax2.axvline(0.5, color="red", linestyle="--")
        st.pyplot(fig2)

        # Results table
        st.subheader("📋 Prediction Results (first 50 rows)")
        st.dataframe(df.head(50))

        # Download
        csv_out = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(csv_out.name, index=False)
        with open(csv_out.name, "rb") as f:
            st.download_button("⬇️ Download Results", f, file_name="urban_predictions.csv")
else:
    st.info("⬆️ Upload a CSV to get started.")
# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.header("📦 Model Information")
if model_data:
    feature_list = model_data.get("feature_names", [])
    st.sidebar.write(f"Algorithm: {type(model_data['model']).__name__}")
    st.sidebar.write(f"Features: {len(feature_list) if feature_list else 'N/A'}")
    if feature_list:
        st.sidebar.write("Expected features:")
        st.sidebar.write(feature_list)
else:
    st.sidebar.warning("⚠️ No model loaded.")
