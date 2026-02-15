import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ------------------------------
# Page Title
# ------------------------------
st.title("Multiclass Classification Model Evaluation App - Dry Bean Dataset")

st.write("Upload test dataset (CSV) and evaluate selected model.")
st.write("The last column of the uploaded dataset should be the target variable.")    
st.write("Models available: Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes (Gaussian), Random Forest (Ensemble), XGBoost/Gradient Boosting (Ensemble)")   
st.write("Evaluation metrics include Accuracy, Precision, Recall, F1 Score, MCC, AUC Score, Confusion Matrix, and Classification Report.")  
st.write("Ensure that the uploaded dataset has the same feature columns as the training data used for these models.")   


# ------------------------------
# Model Selection
# ------------------------------
model_option = st.selectbox(
    "Select Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes (Gaussian)",
        "Random Forest (Ensemble)",
        "XGBoost/Gradient Boosting (Ensemble)"
    )
)

# ------------------------------
# Load Selected Model
# ------------------------------
def load_model(model_name):
    if model_name == "Logistic Regression":
        return joblib.load("models/logistic_regression.pkl")
    elif model_name == "Decision Tree":
        return joblib.load("models/decision_tree.pkl")
    elif model_name == "K-Nearest Neighbors":
        return joblib.load("models/knn.pkl")
    elif model_name == "Naive Bayes (Gaussian)":
        return joblib.load("models/naive_bayes.pkl")
    elif model_name == "Random Forest (Ensemble)":
        return joblib.load("models/random_forest.pkl")
    elif model_name == "XGBoost/Gradient Boosting (Ensemble)":
        return joblib.load("models/xgboost.pkl")

# ------------------------------
# File Upload
# ------------------------------
st.subheader("Upload Test Dataset")
st.write("Please upload a CSV file containing the test dataset. The last column should be the target variable, and the other columns should match the features used during model training.")    
st.write("File available for download: [Test Dataset](https://github.com/SRT85/BITS-ML-AS-2-V1.1/blob/main/data/test.csv)")
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.write(df.head())

    #last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = load_model(model_option)

    # Prediction
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    # ------------------------------
    # Evaluation Metrics
    # ------------------------------
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mcc = matthews_corrcoef(y, y_pred)
    auc = roc_auc_score(y, y_prob, multi_class='ovr')

    st.subheader("Evaluation Metrics")

    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"MCC: {mcc:.4f}")
    st.write(f"AUC Score: {auc:.4f}")

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    st.subheader("Confusion Matrix")

    #cm = confusion_matrix(y, y_pred)
    #st.write(cm)
    labels = sorted(set(y))

    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    #st.subheader("Confusion Matrix")
    st.dataframe(cm_df)
    # ------------------------------
    # Classification Report
    # ------------------------------
    #st.subheader("Classification Report")
    #report = classification_report(y, y_pred)
    #st.text(report)
    
    st.subheader("Classification Report")   
    report_dict = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    st.subheader("Thank you! Model Evaluation Completed")