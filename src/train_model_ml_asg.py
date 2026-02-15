import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score

#from xgboost import XGBClassifier

# ---------------------------------
# Create models directory
# ---------------------------------
print("Creating models directory...")
os.makedirs("models", exist_ok=True)

# ---------------------------------
# Load Dry Bean Dataset
# ---------------------------------
print("Loading dataset...")
df = pd.read_excel("data/Dry_Bean_Dataset.xlsx")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("Splitting dataset into train and test sets...")
# ---------------------------------
# Train-Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train-test split done.")
# ---------------------------------
# Logistic Regression
# ---------------------------------
logistic_pipeline = Pipeline([
    ('variance', VarianceThreshold()),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=2000))
])

logistic_pipeline.fit(X_train, y_train)
joblib.dump(logistic_pipeline, "models/logistic_regression.pkl")

print("Logistic Regression saved")


# ---------------------------------
# Decision Tree
# ---------------------------------
dt_pipeline = Pipeline([
    ('model', DecisionTreeClassifier(random_state=42))
])

dt_pipeline.fit(X_train, y_train)
joblib.dump(dt_pipeline, "models/decision_tree.pkl")

print("Decision Tree saved")


# ---------------------------------
# KNN
# ---------------------------------
knn_pipeline = Pipeline([
    ('variance', VarianceThreshold()),
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)
joblib.dump(knn_pipeline, "models/knn.pkl")

print("KNN saved")


# ---------------------------------
# Gaussian Naive Bayes
# ---------------------------------
nb_pipeline = Pipeline([
    ('variance', VarianceThreshold()),
    ('scaler', StandardScaler()),
    ('model', GaussianNB())
])

nb_pipeline.fit(X_train, y_train)
joblib.dump(nb_pipeline, "models/naive_bayes.pkl")

print("Naive Bayes saved")


# ---------------------------------
# Random Forest
# ---------------------------------
rf_pipeline = Pipeline([
    ('model', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
joblib.dump(rf_pipeline, "models/random_forest.pkl")

print("Random Forest saved")


# ---------------------------------
# Gradient Boosting
# ---------------------------------
xgb_pipeline = Pipeline([
   ('model', GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ))
])

xgb_pipeline.fit(X_train, y_train)
joblib.dump(xgb_pipeline, "models/xgboost.pkl")

#print("XGBoost saved")

# Create XGBoost pipeline
# Could not use XGBoost due to some issues with the library. 
# Will try to fix it later and add it to the project.
#xgb_pipeline = Pipeline([
#    ('scaler', StandardScaler()),
#    ('model', XGBClassifier(
#        n_estimators=200,
#        learning_rate=0.1,
#        max_depth=6,
#        objective='multi:softprob',   # Multi-class
#        eval_metric='mlogloss',
#        random_state=42
#    ))
#])

# Train
#xgb_pipeline.fit(X_train, y_train)

# Save model
#joblib.dump(xgb_pipeline, "models/xgboost.pkl")  

print("\nAll 6 models trained and saved successfully.")

#Generate metrics for all models and save to a CSV file
results = []
for name, model in [
    ("Logistic Regression", logistic_pipeline),
    ("Decision Tree", dt_pipeline),
    ("KNN", knn_pipeline),
    ("Naive Bayes", nb_pipeline),
    ("Random Forest", rf_pipeline),
    ("XGBoost", xgb_pipeline)
]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}") 
    print(f"F1 Score: {f1:.4f}")
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "ROC AUC": roc_auc
    })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("data/model_metrics.csv", index=False)
print("Model metrics saved to model_metrics.csv")

print("\nAll models trained, saved, and evaluated successfully.")
