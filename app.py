import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# 1. FIND PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. MODEL PATH
# ============================================================

MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"


# ============================================================
# 3. DATASET PATH
# ============================================================

DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


# ============================================================
# 4. LOAD TRAINED MODEL PACKAGE
# ============================================================

try:

    package = joblib.load(MODEL_PATH)

    model = package["model"]
    scaler = package["scaler"]
    threshold = float(package["threshold"])
    features = package["features"]

except Exception as e:

    st.error(f"❌ Could not load the trained model: {e}")
    st.stop()


# ============================================================
# 5. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# 6. TITLE
# ============================================================

# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    ### AI-Powered Transaction Risk Analysis

    Detect potentially fraudulent credit card transactions using
    an optimized **XGBoost machine learning model**.
    """
)

st.caption(
    "🔐 Machine Learning • ⚡ Real-Time Prediction • 📊 Risk Analysis"
)

st.divider()

# ============================================================
# PROJECT SUMMARY
# ============================================================

summary_col1, summary_col2, summary_col3 = st.columns(3)


with summary_col1:

    st.info(
        """
        **🤖 Model**

        XGBoost
        """
    )


with summary_col2:

    st.info(
        """
        **🎯 PR-AUC**

        0.8272
        """
    )


with summary_col3:

    st.info(
        """
        **⚖️ Fraud Rate**

        0.17%
        """
    )


# ============================================================
# 7. MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "XGBoost"
    )

with col2:
    st.metric(
        "Features",
        len(features)
    )

with col3:
    st.metric(
        "Decision Threshold",
        f"{threshold:.2f}"
    )


st.divider()


# ============================================================
# 8. INTRODUCTION
# ============================================================

st.header("🔍 Transaction Analysis")

st.write(
    "Enter the transaction features below. "
    "The trained XGBoost model will calculate "
    "the probability of fraud."
)


# ============================================================
# 9. QUICK DEMO
# ============================================================

st.subheader("🧪 Quick Demo")

st.write(
    "Instead of entering 30 values manually, "
    "you can load an actual transaction from the dataset."
)


sample = None


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Load genuine transaction
# ------------------------------------------------------------

with col1:
    if st.button(
        "🟢 Load Genuine Transaction",
        use_container_width=True
    ):

        if DATA_PATH.exists():

            try:

                data = pd.read_csv(DATA_PATH)

                genuine_data = data[
                    data["Class"] == 0
                ]

                if len(genuine_data) > 0:

                    sample = genuine_data.iloc[0].copy()
                    st.session_state["sample"] = sample

                    st.success(
                    "🟢 Genuine transaction loaded from dataset."
                    )

            except Exception as e:

                st.error(
                    f"Could not load genuine transaction: {e}"
                )

        else:

            st.warning(
                "⚠️ Dataset is not available online. "
                "Please enter the transaction values manually."
            )
    

# ------------------------------------------------------------
# Load fraud transaction
# ------------------------------------------------------------

with col2:
    if st.button(
    "🔴 Load Fraud Transaction",
    use_container_width=True
):

        if DATA_PATH.exists():

            try:

                data = pd.read_csv(DATA_PATH)

                fraud_data = data[
                    data["Class"] == 1
                ]

                if len(fraud_data) > 0:

                    sample = fraud_data.iloc[0].copy()
                    st.session_state["sample"] = sample

                    st.error(
                        "🔴 Fraud transaction loaded from dataset."
                    )

            except Exception as e:

                st.error(
                    f"Could not load fraud transaction: {e}"
                )

        else:

            st.warning(
                "⚠️ Dataset is not available online. "
                "Please enter the transaction values manually."
            )

    


# ============================================================
# 10. GET SAVED SAMPLE
# ============================================================

if "sample" in st.session_state:

    sample = st.session_state["sample"]


# ============================================================
# 11. INPUT SECTION
# ============================================================

st.subheader("💰 Basic Transaction Details")

inputs = {}


# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

default_time = 0.0

if sample is not None:
    default_time = float(sample["Time"])


inputs["Time"] = st.number_input(
    "Transaction Time",
    value=default_time,
    format="%.6f",
    help="Transaction time in seconds."
)


# ------------------------------------------------------------
# Amount
# ------------------------------------------------------------

default_amount = 0.0

if sample is not None:
    default_amount = float(sample["Amount"])


inputs["Amount"] = st.number_input(
    "Transaction Amount",
    value=default_amount,
    min_value=0.0,
    format="%.2f",
    help="Transaction amount."
)


# ============================================================
# 12. ADVANCED FEATURES
# ============================================================




# ------------------------------------------------------------
# V1 to V28
# ------------------------------------------------------------
# ============================================================
# V1 - V28 FEATURES IN TWO COLUMNS
# ============================================================

st.subheader("🧠 Advanced Transaction Features")

st.write(
    "V1–V28 are anonymized transaction features "
    "used by the trained XGBoost model."
)

# Create two columns
left_column, right_column = st.columns(2)

# Get only V1 to V28
advanced_features = [
    feature
    for feature in features
    if feature not in ["Time", "Amount"]
]

# Split features into two groups
middle = len(advanced_features) // 2

left_features = advanced_features[:middle]
right_features = advanced_features[middle:]


# ============================================================
# LEFT COLUMN — V1 to V14
# ============================================================

with left_column:

    for feature in left_features:

        default_value = 0.0

        if sample is not None and feature in sample:

            try:
                default_value = float(sample[feature])

            except:
                default_value = 0.0

        inputs[feature] = st.number_input(
            feature,
            value=default_value,
            format="%.6f"
        )


# ============================================================
# RIGHT COLUMN — V15 to V28
# ============================================================

with right_column:

    for feature in right_features:

        default_value = 0.0

        if sample is not None and feature in sample:

            try:
                default_value = float(sample[feature])

            except:
                default_value = 0.0

        inputs[feature] = st.number_input(
            feature,
            value=default_value,
            format="%.6f"
        )


st.divider()


# ============================================================
# 13. ANALYZE BUTTON
# ============================================================

predict_button = st.button(
    "🔎 Analyze Transaction",
    use_container_width=True,
    type="primary"
)


# ============================================================
# 14. PREDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # STEP A — CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame(
            [[inputs[feature] for feature in features]],
            columns=features
        )


        # ====================================================
        # STEP B — CHECK INPUT SHAPE
        # ====================================================

        if input_data.shape[1] != len(features):

            st.error(
                f"Expected {len(features)} features "
                f"but received {input_data.shape[1]}."
            )

            st.stop()


        # ====================================================
        # STEP C — SCALE DATA CORRECTLY
        # ====================================================

        # IMPORTANT:
        # Your saved scaler expects 2 features:
        # Time and Amount.
        #
        # Therefore we scale ONLY Time and Amount.
        #
        # V1-V28 are NOT scaled here.

        if scaler.n_features_in_ == 2:

            scale_columns = [
                "Time",
                "Amount"
            ]

            scaled_two = scaler.transform(
                input_data[scale_columns].to_numpy()
            )

            # Put scaled values back into the complete
            # 30-feature dataframe.

            model_input = input_data.copy()

            model_input["Time"] = scaled_two[0][0]

            model_input["Amount"] = scaled_two[0][1]


        # ====================================================
        # IF SCALER EXPECTS ALL 30 FEATURES
        # ====================================================

        elif scaler.n_features_in_ == len(features):

            scaled_values = scaler.transform(
                input_data
            )

            model_input = pd.DataFrame(
                scaled_values,
                columns=features
            )


        # ====================================================
        # UNKNOWN SCALER
        # ====================================================

        else:

            st.error(
                f"❌ Scaler expects "
                f"{scaler.n_features_in_} features, "
                f"but the application has "
                f"{len(features)} features."
            )

            st.stop()


        # ====================================================
        # STEP D — ENSURE CORRECT FEATURE ORDER
        # ====================================================

        model_input = model_input[
            features
        ]


        # ====================================================
        # STEP E — GET FRAUD PROBABILITY
        # ====================================================

        probability = float(
            model.predict_proba(
                model_input
            )[0][1]
        )


        # ====================================================
        # STEP F — APPLY OPTIMIZED THRESHOLD
        # ====================================================

        prediction = (
            probability >= threshold
        )


        # ====================================================
        # STEP G — DISPLAY RESULT
        # ====================================================

        st.divider()

        st.header("📊 Prediction Result")
        st.caption(
            "Prediction is based on the optimized XGBoost decision threshold."
        )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability_percentage = (
            probability * 100
        )

        st.metric(
            "Fraud Probability",
            f"{probability_percentage:.6f}%"
        )


        # ----------------------------------------------------
        # Progress bar
        # ----------------------------------------------------

        st.progress(
            float(
                max(
                    0.0,
                    min(
                        probability,
                        1.0
                    )
                )
            )
        )


        # ====================================================
        # FRAUD RESULT
        # ====================================================

        if prediction:

            st.error(
                "🔴 FRAUDULENT TRANSACTION"
            )

            st.warning(
                "This transaction has been classified "
                "as potentially fraudulent."
            )

            risk_level = "HIGH"


        # ====================================================
        # GENUINE RESULT
        # ====================================================

        else:

            st.success(
                "🟢 GENUINE TRANSACTION"
            )

            st.info(
                "This transaction has been classified "
                "as genuine."
            )

            risk_level = "LOW"


        # ====================================================
        # RESULT DETAILS
        # ====================================================

        st.subheader("📈 Risk Analysis")

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability_percentage:.6f}%"
            )


        with col2:

            st.metric(
                "Risk Level",
                risk_level
            )


        with col3:

            st.metric(
                "Decision Threshold",
                f"{threshold:.2%}"
            )


        # ====================================================
        # TECHNICAL INFORMATION
        # ====================================================

        with st.expander(
            "🔧 Technical Prediction Details"
        ):

            st.write(
                "Model: XGBoost"
            )

            st.write(
                f"Number of features: {len(features)}"
            )

            st.write(
                f"Fraud probability: {probability:.6f}"
            )

            st.write(
                f"Decision threshold: {threshold:.6f}"
            )

            st.write(
                f"Prediction: "
                f"{'Fraud' if prediction else 'Genuine'}"
            )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            f"❌ Prediction error: {e}"

        )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()

st.header("📊 Model Performance")

st.write(
    "The following metrics show how the machine learning "
    "models performed during evaluation."
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("🏆 Model Comparison")

# PR-AUC scores obtained during model evaluation
model_comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "PR-AUC": [
            0.6719,
            0.8012,
            0.8272
        ]
    }
)


# Display the comparison table
st.dataframe(
    model_comparison,
    use_container_width=True,
    hide_index=True
)


# Display a bar chart
# ============================================================
# MODEL COMPARISON CHART
# ============================================================

# ============================================================
# MODEL COMPARISON - PR-AUC
# ============================================================

st.subheader("🏆 Model Comparison")

# Model names and PR-AUC scores
models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

pr_auc_scores = [
    0.6719,
    0.8012,
    0.8272
]

# Create a smaller figure
fig, ax = plt.subplots(
    figsize=(5, 3),
    dpi=100
)

# Create bars
bars = ax.bar(
    models,
    pr_auc_scores
)

# Add values above bars
for bar, score in zip(bars, pr_auc_scores):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{score:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Chart title
ax.set_title(
    "Model Comparison - PR-AUC",
    fontsize=12
)

# Axis labels
ax.set_xlabel(
    "Model",
    fontsize=9
)

ax.set_ylabel(
    "PR-AUC",
    fontsize=9
)

# Smaller tick labels
ax.tick_params(
    axis="both",
    labelsize=8
)

# Rotate model names slightly
plt.xticks(
    rotation=15,
    ha="right"
)

# Keep y-axis between 0 and 1
ax.set_ylim(
    0,
    1.0
)

# Improve spacing
fig.tight_layout()

# Display smaller chart
st.pyplot(
    fig,
    use_container_width=False
)

# Close figure
plt.close(fig)

# ============================================================
# FINAL XGBOOST PERFORMANCE
# ============================================================

st.subheader("🚀 Final XGBoost Performance")


# Create three columns
col1,col2, col3 = st.columns(3)


with col1:

    st.metric(
        "PR-AUC",
        "0.8272"
    )


with col2:

    st.metric(
        "Optimal Threshold",
        "0.9302"
    )


with col3:

    st.metric(
        "Validation F1",
        "0.9078"
    )


# ============================================================
# TEST PERFORMANCE
# ============================================================

# ============================================================
# FINAL TEST PERFORMANCE
# ============================================================

st.subheader("🧪 Final Test Performance")

st.write(
    "The following metrics summarize the performance of the "
    "final XGBoost model on the unseen test dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Precision",
        "0.99"
    )

with col2:
    st.metric(
        "Recall",
        "0.75"
    )

with col3:
    st.metric(
        "F1-Score",
        "0.85"
    )
st.markdown("### 📌 What These Results Mean")

st.markdown(
    """
    **Precision (0.99):** When the model predicts a transaction as
    fraudulent, it is correct about 99% of the time.

    **Recall (0.75):** The model successfully detects about 75% of
    the fraudulent transactions in the test set.

    **F1-Score (0.85):** This balances precision and recall and gives
    a useful overall measure of fraud detection performance.
    """
)





# ============================================================
# CONFUSION MATRIX
# ============================================================

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔢 Confusion Matrix")

# Final XGBoost confusion matrix
cm = np.array([
    [56650, 1],
    [24, 71]
])

# Create same size as Model Comparison
fig, ax = plt.subplots(
    figsize=(5, 3),
    dpi=100
)

# Display confusion matrix
image = ax.imshow(cm)

# Add values inside each cell
for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            fontsize=10
        )

# Axis labels
ax.set_xlabel(
    "Predicted Class",
    fontsize=9
)

ax.set_ylabel(
    "Actual Class",
    fontsize=9
)

# Tick labels
ax.set_xticks([0, 1])
ax.set_xticklabels(
    ["Genuine", "Fraud"],
    fontsize=8
)

ax.set_yticks([0, 1])
ax.set_yticklabels(
    ["Genuine", "Fraud"],
    fontsize=8
)

# Title
ax.set_title(
    "Final XGBoost Confusion Matrix",
    fontsize=12
)

# Add color scale
fig.colorbar(
    image,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

# Improve spacing
fig.tight_layout()

# Display same compact size
st.pyplot(
    fig,
    use_container_width=False
)

# Close figure
plt.close(fig)
# ============================================================
# EXPLANATION
# ============================================================

st.subheader("📌 What These Results Mean")

st.write(
    """
    **Precision (0.99):** When the model predicts a transaction
    as fraudulent, it is correct about 99% of the time.

    **Recall (0.75):** The model successfully detects about 75%
    of the fraudulent transactions in the test set.

    **F1-Score (0.85):** This balances precision and recall and
    provides a useful overall measure for fraud detection.

    **PR-AUC (0.8272):** This is particularly useful for this
    project because fraud transactions are highly imbalanced
    compared with genuine transactions.

    **Optimal Threshold (0.9302):** Instead of using the default
    0.50 probability threshold, the threshold was optimized using
    validation data to improve the fraud classification balance.
    """
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("🔎 Feature Importance")

# Get feature importance from the trained XGBoost model
importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# Show top 10 features
top_features = importance_df.head(10)

# Create same compact figure size
fig, ax = plt.subplots(
    figsize=(5, 3),
    dpi=100
)

# Horizontal bar chart
ax.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

# Title
ax.set_title(
    "Top 10 Feature Importance",
    fontsize=12
)

# Axis labels
ax.set_xlabel(
    "Importance",
    fontsize=9
)

ax.set_ylabel(
    "Feature",
    fontsize=9
)

# Smaller tick labels
ax.tick_params(
    axis="both",
    labelsize=8
)

# Improve spacing
fig.tight_layout()

# Display same compact size
st.pyplot(
    fig,
    use_container_width=False
)

# Close figure
plt.close(fig)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()

st.header("📊 Dataset Overview")

st.write(
    "The statistics below represent the cleaned credit card "
    "transaction dataset used for the fraud detection model."
)


# ============================================================
# LOAD DATASET
# ============================================================

if DATA_PATH.exists():

    try:

        # Load the original CSV dataset
        dataset = pd.read_csv(DATA_PATH)

        # Remove duplicate transactions
        dataset = dataset.drop_duplicates()

        # Calculate transaction counts
        total_transactions = len(dataset)

        genuine_transactions = int(
            (dataset["Class"] == 0).sum()
        )

        fraud_transactions = int(
            (dataset["Class"] == 1).sum()
        )

        fraud_percentage = (
            fraud_transactions /
            total_transactions
        ) * 100

    except Exception as e:

        st.error(
            f"Unable to load dataset statistics: {e}"
        )

        total_transactions = 283726
        genuine_transactions = 283253
        fraud_transactions = 473
        fraud_percentage = 0.17

else:

    # Dataset is intentionally excluded from GitHub
    # because of its large file size.

    total_transactions = 283726
    genuine_transactions = 283253
    fraud_transactions = 473
    fraud_percentage = 0.17


# ============================================================
# DISPLAY DATASET STATISTICS
# ============================================================

st.subheader("📈 Transaction Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:

    st.metric(
        "Genuine Transactions",
        f"{genuine_transactions:,}"
    )

with col3:

    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )

with col4:

    st.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.2f}%"
    )


# ============================================================
# TRANSACTION CLASS DISTRIBUTION
# ============================================================

st.subheader(
    "📊 Transaction Class Distribution"
)

distribution = pd.DataFrame(
    {
        "Class": [
            "Genuine",
            "Fraud"
        ],
        "Transactions": [
            genuine_transactions,
            fraud_transactions
        ]
    }
)

st.bar_chart(
    distribution.set_index("Class")
)


# ============================================================
# IMBALANCE INFORMATION
# ============================================================

st.info(
    f"""
    The dataset is highly imbalanced. There are
    **{fraud_transactions:,} fraudulent transactions**
    compared with **{genuine_transactions:,} genuine
    transactions**.

    Fraud represents only **{fraud_percentage:.2f}%**
    of the cleaned dataset. This is why the project
    uses class weighting, PR-AUC, and threshold
    optimization instead of relying only on accuracy.
    """
)

# ============================================================
# ABOUT THIS PROJECT
# ============================================================

st.divider()

st.header("ℹ️ About This Project")

st.write(
    """
    This Credit Card Fraud Detection system uses machine learning
    to identify potentially fraudulent transactions.

    The project addresses a highly imbalanced classification
    problem where fraudulent transactions represent only a small
    fraction of the overall dataset.
    """
)


# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.subheader("🔄 Machine Learning Workflow")

st.markdown(
    """
    **1. Data Preparation**  
    Duplicate transactions were removed and the dataset was
    separated into features and target labels.

    **2. Preprocessing**  
    Numerical features were standardized using the same scaler
    during both training and prediction.

    **3. Model Comparison**  
    Logistic Regression, Random Forest, and XGBoost were evaluated
    using PR-AUC.

    **4. Imbalance Handling**  
    Class weighting was used to give greater importance to the
    minority fraud class.

    **5. Threshold Optimization**  
    The classification threshold was optimized using validation
    data rather than relying on the default 0.50 threshold.

    **6. Final Model**  
    XGBoost achieved the strongest PR-AUC and was selected as
    the final fraud detection model.

    **7. Deployment**  
    The trained model was integrated into a Streamlit application
    for interactive transaction-level predictions.
    """
)


# ============================================================
# TECHNOLOGIES USED
# ============================================================

st.subheader("🛠️ Technologies Used")

tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)


with tech_col1:

    st.info("🐍 Python")


with tech_col2:

    st.info("📊 Pandas")


with tech_col3:

    st.info("🤖 XGBoost")


with tech_col4:

    st.info("🌐 Streamlit")


# ============================================================
# KEY PROJECT HIGHLIGHTS
# ============================================================

st.subheader("⭐ Key Highlights")

st.markdown(
    """
    - Highly imbalanced fraud classification problem
    - Duplicate transaction removal
    - Feature standardization
    - Comparison of multiple machine learning models
    - Class imbalance handling using class weighting
    - PR-AUC based model evaluation
    - Validation-based threshold optimization
    - Interactive fraud probability prediction
    - Feature importance visualization
    - Confusion matrix analysis
    """
)