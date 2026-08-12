import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px;
    padding: 12px;
}
.card {
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 16px;
}
.small-note {font-size: 0.88rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("best_model.joblib")
SCALER_PATH = Path("scaler.joblib")
COLUMNS_PATH = Path("model_columns.joblib")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Missing {SCALER_PATH}")
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Missing {COLUMNS_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_columns = list(joblib.load(COLUMNS_PATH))
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except Exception as e:
    st.error("The deployment artifacts could not be loaded.")
    st.exception(e)
    st.stop()

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection")
st.caption(
    "Interactive machine-learning application for estimating whether a "
    "credit-card transaction is potentially fraudulent."
)

st.info(
    "Enter the transaction values below. The application applies the same "
    "feature order and scaling used during model development before prediction."
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("Model Information")
    st.metric("Model", type(model).__name__)
    st.metric("Input features", len(model_columns))
    st.metric("Scaler features", getattr(scaler, "n_features_in_", "N/A"))
    st.divider()
    st.markdown(
        "**Important:** This application is a machine-learning prediction "
        "tool and should not be used as the sole basis for approving or "
        "declining a financial transaction."
    )

# The notebook uses exactly these 30 predictors:
# Time, V1...V28, Amount.
expected = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Keep deployment safe if the serialized column list differs.
if set(model_columns) != set(expected):
    st.warning(
        "The saved model column schema differs from the schema documented "
        "in the supplied notebook. Inputs will follow the serialized model "
        "column order."
    )

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
manual_tab, batch_tab, about_tab = st.tabs(
    ["🔎 Single Transaction", "📄 Batch CSV", "ℹ️ About"]
)

def prepare_input(df):
    """Validate, order and scale an inference dataframe."""
    missing = [c for c in model_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in model_columns]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    x = df[model_columns].copy()

    # Ensure all values are numeric.
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if x.isna().any().any():
        bad = x.columns[x.isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values found in: {bad}")

    x_scaled = scaler.transform(x)
    return x, x_scaled, extra

def prediction_table(pred, proba=None):
    result = pd.DataFrame({"Prediction": pred})
    if proba is not None:
        classes = list(model.classes_)
        for i, cls in enumerate(classes):
            result[f"Probability_{cls}"] = np.round(proba[:, i] * 100, 2)
    return result

with manual_tab:
    st.subheader("Enter transaction details")

    st.markdown(
        '<div class="small-note">The dataset represents transactions using '
        'Time, V1–V28 and Amount. V1–V28 are anonymized PCA-derived variables '
        'from the original credit-card dataset.</div>',
        unsafe_allow_html=True,
    )

    time_value = st.number_input(
        "Time",
        min_value=0.0,
        value=0.0,
        step=1.0,
        help="Elapsed time in seconds relative to the first transaction in the dataset."
    )

    amount = st.number_input(
        "Amount",
        min_value=0.0,
        value=100.0,
        step=1.0,
        help="Transaction amount before the training-time log1p transformation."
    )

    with st.expander("Advanced transaction features (V1–V28)", expanded=False):
        values = {}
        cols = st.columns(4)
        for i in range(1, 29):
            with cols[(i - 1) % 4]:
                values[f"V{i}"] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.8f",
                    key=f"v{i}",
                )

    if st.button(
        "🚨 Analyze Transaction",
        type="primary",
        use_container_width=True,
    ):
        # IMPORTANT:
        # The notebook applies log1p to Amount before fitting the model.
        input_row = {"Time": time_value, **values, "Amount": np.log1p(amount)}
        df = pd.DataFrame([input_row])

        try:
            _, x_scaled, _ = prepare_input(df)
            pred = model.predict(x_scaled)

            proba = model.predict_proba(x_scaled) if hasattr(model, "predict_proba") else None
            label = pred[0]

            # The supplied notebook uses Class 0/1.
            if str(label) == "1" or label == 1:
                st.error("⚠️ Prediction: POTENTIALLY FRAUDULENT")
                st.warning(
                    "The model classified this transaction as class 1. "
                    "This is a model prediction, not a definitive fraud determination."
                )
            else:
                st.success("✅ Prediction: LIKELY LEGITIMATE")

            if proba is not None:
                classes = list(model.classes_)
                prob_map = {
                    str(cls): float(proba[0, i]) * 100
                    for i, cls in enumerate(classes)
                }

                c1, c2 = st.columns(2)
                if "0" in prob_map:
                    c1.metric("Class 0 probability", f"{prob_map['0']:.2f}%")
                if "1" in prob_map:
                    c2.metric("Class 1 probability", f"{prob_map['1']:.2f}%")

                st.progress(
                    min(max(prob_map.get("1", 0.0) / 100, 0.0), 1.0),
                    text=f"Fraud probability: {prob_map.get('1', 0.0):.2f}%"
                )

        except Exception as e:
            st.error("Prediction could not be completed.")
            st.exception(e)

with batch_tab:
    st.subheader("Batch prediction")
    st.write(
        "Upload a CSV containing the same predictor columns used by the model. "
        "The target column `Class` is not required."
    )

    template = pd.DataFrame(columns=expected)
    st.download_button(
        "⬇️ Download CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="credit_card_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            st.write("Preview", batch.head())

            missing = [c for c in model_columns if c not in batch.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                x = batch[model_columns].copy()

                # The training notebook applied log1p to Amount.
                if "Amount" in x.columns:
                    x["Amount"] = np.log1p(
                        pd.to_numeric(x["Amount"], errors="coerce")
                    )

                _, x_scaled, _ = prepare_input(x)

                pred = model.predict(x_scaled)
                proba = model.predict_proba(x_scaled) if hasattr(model, "predict_proba") else None

                results = batch.copy()
                results["Predicted_Class"] = pred

                if proba is not None and 1 in model.classes_:
                    fraud_idx = list(model.classes_).index(1)
                    results["Fraud_Probability_%"] = np.round(
                        proba[:, fraud_idx] * 100, 2
                    )

                st.success(f"Processed {len(results):,} transaction(s).")
                st.dataframe(results, use_container_width=True)

                st.download_button(
                    "⬇️ Download predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error("The uploaded CSV could not be processed.")
            st.exception(e)

with about_tab:
    st.subheader("About this model")
    st.markdown(
        """
        **Project:** Capstone Project 2 – Credit Card Fraud Prediction

        **Target:** `Class`

        - `0` = legitimate transaction
        - `1` = fraudulent transaction

        **Predictors:** `Time`, `V1`–`V28`, and `Amount`.

        The training workflow in the supplied notebook:
        1. Removes duplicate records.
        2. Applies `log1p` to `Amount`.
        3. Splits the data into training and testing sets.
        4. Fits `StandardScaler` on the training data.
        5. Applies SMOTE to the scaled training data only.
        6. Trains multiple classification algorithms.
        7. Evaluates the models using classification reports, confusion matrices and ROC-AUC.
        8. Saves the selected model, scaler and feature-column list for deployment.
        """
    )
    st.caption(
        "For financial decision-making, predictions should be combined with "
        "appropriate fraud-monitoring rules, human review and transaction-risk controls."
    )
