import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "sustainability_training_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "sustainability_model.joblib"
)


# -----------------------------
# Load Dataset
# -----------------------------

print("=" * 60)
print("EcoSense AI - Model Training")
print("=" * 60)

print("\nLoading dataset...")

data = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")

print("\nDataset columns:")
print(list(data.columns))


# -----------------------------
# Feature Configuration
# -----------------------------

FEATURES = [
    "electricity",
    "transport",
    "waste",
    "water"
]


# -----------------------------
# Validate Dataset
# -----------------------------

missing_features = [
    feature
    for feature in FEATURES
    if feature not in data.columns
]

if missing_features:
    raise ValueError(
        f"Missing required feature columns: {missing_features}"
    )


# Find target column
possible_targets = [
    "sustainability",
    "category",
    "label",
    "target",
    "classification"
]

target_column = None

for column in possible_targets:
    if column in data.columns:
        target_column = column
        break


if target_column is None:

    remaining_columns = [
        column
        for column in data.columns
        if column not in FEATURES
    ]

    if len(remaining_columns) == 1:
        target_column = remaining_columns[0]

    else:
        raise ValueError(
            "Could not automatically identify the target column. "
            f"Available columns: {list(data.columns)}"
        )


print(f"\nTarget column: {target_column}")


# -----------------------------
# Data Cleaning
# -----------------------------

print("\nChecking missing values...")

missing_values = data[FEATURES + [target_column]].isnull().sum()

print(missing_values)


if missing_values.sum() > 0:

    print("\nRemoving rows containing missing values...")

    data = data.dropna(
        subset=FEATURES + [target_column]
    )


# Remove duplicate rows
duplicate_count = data.duplicated().sum()

if duplicate_count > 0:

    print(
        f"\nRemoving {duplicate_count} duplicate rows..."
    )

    data = data.drop_duplicates()


# -----------------------------
# Feature / Target Split
# -----------------------------

X = data[FEATURES].copy()
y = data[target_column].copy()


# Make sure numerical features are numeric
for feature in FEATURES:

    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )


# Remove rows that became invalid after conversion
valid_rows = X.notnull().all(axis=1)

X = X.loc[valid_rows]
y = y.loc[valid_rows]


print("\nFinal dataset:")
print(f"Samples: {len(X)}")
print(f"Features: {FEATURES}")
print(f"Classes: {sorted(y.astype(str).unique())}")


# -----------------------------
# Train / Test Split
# -----------------------------

print("\nCreating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# -----------------------------
# Model Training
# -----------------------------

print("\nTraining Decision Tree model...")

model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


print("Model training completed.")


# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Model Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------
# Confusion Matrix
# -----------------------------

print("Confusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# -----------------------------
# Feature Importance
# -----------------------------

print("\nFeature Importance:")

feature_importance = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.to_string(
        index=False
    )
)


# -----------------------------
# Save Model
# -----------------------------

joblib.dump(
    model,
    MODEL_PATH
)

print(
    f"\nModel saved successfully to:\n{MODEL_PATH}"
)


# -----------------------------
# Save Evaluation Results
# -----------------------------

evaluation_results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "features": FEATURES
}


EVALUATION_PATH = os.path.join(
    BASE_DIR,
    "src",
    "model_evaluation.joblib"
)

joblib.dump(
    evaluation_results,
    EVALUATION_PATH
)


# -----------------------------
# Final Output
# -----------------------------

print(
    f"Evaluation results saved to:\n{EVALUATION_PATH}"
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)