import os
import random
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# Generate Synthetic Training Data
# -----------------------------------

random.seed(42)

data = []

for _ in range(1000):

    electricity = random.randint(50, 500)
    transport = random.randint(0, 3)
    waste = round(random.uniform(1, 20), 1)
    water = random.randint(50, 350)

    # Sustainability impact score
    impact = (
        electricity * 0.10
        + transport * 12
        + waste * 2
        + water * 0.05
    )

    # Classification
    if impact < 45:
        category = "Excellent"

    elif impact < 65:
        category = "Good"

    elif impact < 90:
        category = "Moderate"

    else:
        category = "Needs Improvement"

    data.append([
        electricity,
        transport,
        waste,
        water,
        category
    ])


# -----------------------------------
# Create DataFrame
# -----------------------------------

df = pd.DataFrame(
    data,
    columns=[
        "electricity",
        "transport",
        "waste",
        "water",
        "category"
    ]
)


# -----------------------------------
# Save Training Dataset
# -----------------------------------

data_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data"
)

os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(
    data_dir,
    "sustainability_training_data.csv"
)

df.to_csv(csv_path, index=False)


# -----------------------------------
# Features and Target
# -----------------------------------

X = df[
    [
        "electricity",
        "transport",
        "waste",
        "water"
    ]
]

y = df["category"]


# -----------------------------------
# Train / Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------
# Train Decision Tree
# -----------------------------------

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)
# -----------------------------------
# Model Evaluation
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print(f"Test samples: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------
# Save Model
# -----------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "sustainability_model.joblib"
)

joblib.dump(model, model_path)


# -----------------------------------
# Training Summary
# -----------------------------------

print("EcoSense AI model trained successfully!")
print(f"Training samples: {len(df)}")
print(f"Dataset saved to: {csv_path}")
print(f"Model saved to: {model_path}")
print("\nCategory distribution:")
print(df["category"].value_counts())