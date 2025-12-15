# ============================================================
# PROJECT: Loan Approval Prediction using
# Discriminant Analysis (LDA & QDA)
# ============================================================

# -------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# 2. LOAD DATASET
# -------------------------------
# Load Loan Approval dataset
data = pd.read_csv("dataset_one.csv")

print("Initial Dataset Shape:", data.shape)

# -------------------------------
# 3. DATA CLEANING & PREPROCESSING
# -------------------------------

# Drop ID column (not useful for prediction)
data = data.drop("Loan_ID", axis=1)

# Numerical columns: fill missing values using mean
numeric_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

# Categorical columns: fill missing values using mode
categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Credit_History"
]

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# Convert '3+' in Dependents to numeric value
data["Dependents"] = data["Dependents"].replace("3+", "3").astype(int)

# Encode categorical variables into numeric values
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Married"] = data["Married"].map({"Yes": 1, "No": 0})
data["Self_Employed"] = data["Self_Employed"].map({"Yes": 1, "No": 0})
data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate": 0})
data["Property_Area"] = data["Property_Area"].map({
    "Urban": 2,
    "Semiurban": 1,
    "Rural": 0
})

# Encode target variable (Loan Status)
data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

print("Dataset Shape After Cleaning:", data.shape)

# -------------------------------
# 4. FEATURES & TARGET
# -------------------------------
X = data.drop("Loan_Status", axis=1)  # Features
y = data["Loan_Status"]               # Target

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y  # keep class distribution balanced
)

# -------------------------------
# 6. LINEAR DISCRIMINANT ANALYSIS (LDA)
# -------------------------------
# Create pipeline: scaling + LDA
lda_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis())
])

# Train LDA model
lda_pipeline.fit(X_train, y_train)

# Predict using LDA
y_pred_lda = lda_pipeline.predict(X_test)

# -------------------------------
# 7. LDA EVALUATION
# -------------------------------
lda_accuracy = accuracy_score(y_test, y_pred_lda)
lda_conf_matrix = confusion_matrix(y_test, y_pred_lda)

print("\n================ LDA RESULTS ================")
print("LDA Accuracy:", lda_accuracy)
print("\nConfusion Matrix:")
print(lda_conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lda))

# -------------------------------
# 8. CROSS-VALIDATION (LDA)
# -------------------------------
# Evaluate model stability using cross-validation
lda_cv_scores = cross_val_score(
    lda_pipeline, X, y,
    cv=10,
    scoring="accuracy"
)

print("\nLDA Cross-Validation Mean Accuracy:", lda_cv_scores.mean())
print("LDA Cross-Validation Std:", lda_cv_scores.std())

# -------------------------------
# 9. QUADRATIC DISCRIMINANT ANALYSIS (QDA)
# -------------------------------
# Create pipeline: scaling + QDA
qda_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("qda", QuadraticDiscriminantAnalysis())
])

# Train QDA model
qda_pipeline.fit(X_train, y_train)

# Predict using QDA
y_pred_qda = qda_pipeline.predict(X_test)

# -------------------------------
# 10. QDA EVALUATION
# -------------------------------
qda_accuracy = accuracy_score(y_test, y_pred_qda)
qda_conf_matrix = confusion_matrix(y_test, y_pred_qda)

print("\n================ QDA RESULTS ================")
print("QDA Accuracy:", qda_accuracy)
print("\nConfusion Matrix:")
print(qda_conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_qda))

# -------------------------------
# 11. CONFUSION MATRIX VISUALIZATION
# -------------------------------
def plot_confusion_matrix(cm, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    classes = ["Rejected (0)", "Approved (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Display values inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.tight_layout()
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(lda_conf_matrix, "Confusion Matrix - LDA")
plot_confusion_matrix(qda_conf_matrix, "Confusion Matrix - QDA")

# -------------------------------
# 12. FINAL COMPARISON
# -------------------------------
print("\n================ FINAL COMPARISON ================")
print(f"LDA Test Accuracy: {lda_accuracy:.4f}")
print(f"QDA Test Accuracy: {qda_accuracy:.4f}")
print(f"LDA CV Accuracy:  {lda_cv_scores.mean():.4f}")
