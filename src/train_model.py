import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# -----------------------------------
# Training Data
# -----------------------------------

data = [
    # electricity, transport, waste, water, category

    [100, 0, 2, 100, "Excellent"],
    [120, 1, 3, 120, "Excellent"],
    [150, 0, 4, 140, "Excellent"],
    [180, 1, 4, 150, "Good"],
    [200, 0, 5, 160, "Good"],

    [220, 2, 6, 170, "Moderate"],
    [250, 2, 7, 180, "Moderate"],
    [280, 3, 8, 190, "Moderate"],
    [300, 2, 9, 200, "Moderate"],

    [320, 3, 10, 210, "Needs Improvement"],
    [350, 3, 12, 220, "Needs Improvement"],
    [400, 3, 15, 250, "Needs Improvement"],
    [450, 3, 18, 280, "Needs Improvement"],
    [500, 3, 20, 300, "Needs Improvement"],
]


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
# Train Model
# -----------------------------------

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X, y)


# -----------------------------------
# Save Model
# -----------------------------------

model_path = os.path.join(
    os.path.dirname(__file__),
    "sustainability_model.joblib"
)

joblib.dump(model, model_path)


print("EcoSense AI model trained successfully!")
print(f"Model saved to: {model_path}")
print(f"Training samples: {len(df)}")