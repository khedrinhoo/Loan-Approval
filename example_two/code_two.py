# ============================================================
# PROJECT: Personal Loan Prediction using
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
data = pd.read_csv("dataset_two.csv")
print("Initial Dataset Shape:", data.shape)

# -------------------------------
# 3. DATA CLEANING & PREPROCESSING
# -------------------------------

# Drop ID and ZIP Code (not useful for prediction)
data = data.drop(["ID", "ZIP Code"], axis=1)

# Handle non-numeric values (like '2/67' or any other invalid values)
data['CCAvg'] = data['CCAvg'].replace({'/': ''}, regex=True).astype(float)

# Check missing values
print("\nMissing Values:\n", data.isnull().sum())

# Fill missing numeric values with mean (if any)
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    data[col] = data[col].fillna(data[col].mean())

print("Dataset Shape After Cleaning:", data.shape)

# -------------------------------
# 4. FEATURES & TARGET
# -------------------------------
X = data.drop("Personal Loan", axis=1)   # Features
y = data["Personal Loan"]                # Target

# -------------------------------
# 5. TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -------------------------------
# 6. LINEAR DISCRIMINANT ANALYSIS (LDA)
# -------------------------------
lda_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lda", LinearDiscriminantAnalysis())
])

lda_pipeline.fit(X_train, y_train)
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
qda_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("qda", QuadraticDiscriminantAnalysis())
])

qda_pipeline.fit(X_train, y_train)
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

    classes = ["No Loan (0)", "Loan (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.tight_layout()
    plt.show()

plot_confusion_matrix(lda_conf_matrix, "Confusion Matrix - LDA")
plot_confusion_matrix(qda_conf_matrix, "Confusion Matrix - QDA")

# -------------------------------
# 12. FINAL COMPARISON
# -------------------------------
print("\n================ FINAL COMPARISON ================")
print(f"LDA Test Accuracy: {lda_accuracy:.4f}")
print(f"QDA Test Accuracy: {qda_accuracy:.4f}")
print(f"LDA CV Accuracy:  {lda_cv_scores.mean():.4f}")
