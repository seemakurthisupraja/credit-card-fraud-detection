# 💳 Credit Card Fraud Detection

An AI-powered machine learning application that detects potentially fraudulent credit card transactions using an optimized XGBoost classification model and provides real-time risk analysis through a Streamlit dashboard.

---

## 📌 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a very small fraction of all transactions.

This project develops an end-to-end fraud detection system that:

- Processes highly imbalanced transaction data
- Compares multiple machine learning models
- Uses XGBoost as the final classification model
- Optimizes the fraud decision threshold
- Evaluates the model using fraud-focused metrics
- Provides real-time transaction predictions
- Displays fraud probability and risk level
- Visualizes model performance and feature importance
- Provides an interactive Streamlit dashboard

---

## 🎯 Problem Statement

The objective is to identify fraudulent credit card transactions while minimizing incorrect fraud alerts.

Because fraudulent transactions are extremely rare compared with genuine transactions, accuracy alone is not an appropriate evaluation metric.

Therefore, this project focuses on:

- Precision
- Recall
- F1-Score
- PR-AUC
- Confusion Matrix
- Decision threshold optimization

---

## 📊 Dataset

The project uses a credit card transaction dataset containing anonymized transaction features.

### Dataset Characteristics

| Statistic | Value |
|---|---:|
| Total transactions | 283,726 |
| Genuine transactions | 283,253 |
| Fraudulent transactions | 473 |
| Fraud percentage | 0.17% |

The dataset is highly imbalanced, with fraudulent transactions representing only approximately **0.17%** of the cleaned dataset.

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

- Loading the transaction dataset
- Removing unnecessary records/features where applicable
- Separating input features and target variable
- Handling the severe class imbalance
- Splitting the dataset into training, validation, and test sets
- Feature scaling where required
- Maintaining consistent feature ordering between training and prediction

The same preprocessing logic is used during inference to ensure that new transactions are presented to the model in the expected format.

---

## 🤖 Models Evaluated

Multiple machine learning algorithms were evaluated before selecting the final model.

### Models

1. Logistic Regression
2. Random Forest
3. XGBoost

### PR-AUC Comparison

| Model | PR-AUC |
|---|---:|
| Logistic Regression | 0.6719 |
| Random Forest | 0.8012 |
| XGBoost | **0.8272** |

XGBoost achieved the highest PR-AUC and was therefore selected as the final model.

---

## 🚀 Final XGBoost Performance

The final XGBoost model achieved the following results on the test data:

| Metric | Score |
|---|---:|
| PR-AUC | **0.8272** |
| Precision | **0.99** |
| Recall | **0.75** |
| F1-Score | **0.85** |
| Optimal Threshold | **0.9302** |

### What the metrics mean

**Precision — 0.99**

When the model predicts a transaction as fraudulent, approximately 99% of those predictions are correct.

**Recall — 0.75**

The model successfully identifies approximately 75% of the fraudulent transactions in the test set.

**F1-Score — 0.85**

The F1-score provides a balance between precision and recall.

**PR-AUC — 0.8272**

PR-AUC is particularly useful for this project because the dataset contains a severe imbalance between genuine and fraudulent transactions.

---

## 🎯 Decision Threshold Optimization

A standard binary classification model commonly uses a probability threshold of 0.50.

For this project, the decision threshold was optimized using validation data.

### Final threshold

```text
0.9302



---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard that provides real-time fraud prediction and model analysis.

### Dashboard Features

- Real-time transaction prediction
- Fraud probability calculation
- Genuine/Fraud classification
- Risk-level assessment
- Optimized decision threshold
- Model performance comparison
- Confusion matrix visualization
- Feature importance visualization
- Dataset statistics
- Technical prediction details

### Prediction Workflow

```text
Transaction Input
       ↓
Input Validation
       ↓
Feature Ordering
       ↓
Data Preprocessing
       ↓
XGBoost Model
       ↓
Fraud Probability
       ↓
Optimized Threshold
       ↓
Risk Classification
       ↓
Prediction Result