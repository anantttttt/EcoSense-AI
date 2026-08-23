# 🌱 EcoSense AI

## AI-Powered Sustainability Decision Support System

EcoSense AI is a machine-learning-based sustainability decision-support
prototype designed to help users understand the potential environmental
impact of everyday consumption patterns.

The system analyzes:

- ⚡ Monthly electricity consumption
- 🚗 Primary transportation mode
- ♻️ Household waste generation
- 💧 Daily water consumption

It then uses a Decision Tree classification model to classify the user's
overall sustainability profile and provides understandable decision
factors, impact areas, and practical recommendations.

---

## 🎯 Project Objective

The objective of EcoSense AI is to make sustainability assessment more
accessible by converting everyday resource-use information into an
easy-to-understand sustainability assessment.

---

## 🌍 Sustainable Development Goal

### Primary SDG: SDG 12 - Responsible Consumption and Production

EcoSense AI supports responsible consumption by helping users identify
resource-use patterns that may have opportunities for improvement.

---

## 🤖 How AI Is Used

EcoSense AI uses a Decision Tree machine-learning classifier.

### Workflow

User Input
↓
Data Processing
↓
Decision Tree ML Model
↓
Sustainability Classification
↓
Decision Factors
↓
Impact Analysis
↓
Recommendations

### Classification Categories

- Excellent
- Good
- Moderate
- Needs Improvement

---

## 📊 Features

### Sustainability Assessment

Users provide four sustainability-related inputs.

### AI Classification

A trained Decision Tree model classifies the sustainability profile.

### Explainable Results

The application identifies major factors associated with the assessment.

### Impact Analysis

The system highlights potential areas of environmental impact.

### Visual Dashboard

The application displays the user's input profile through a chart.

### Responsible AI

The project considers:

- Fairness
- Transparency
- Ethics
- Privacy

---

## 🧠 Dataset

The current prototype uses a synthetically generated dataset containing
1,000 sustainability profiles.

The dataset is intended for prototype development and demonstration.

It should not be interpreted as representative real-world environmental
survey data.

Future versions should validate the model using representative
real-world datasets.

---

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Git / GitHub

---

## 📁 Project Structure

```text
EcoSense-AI/
│
├── app/
│   └── app.py
│
├── data/
│   └── sustainability_training_data.csv
│
├── docs/
│   └── project_documentation.md
│
├── images/
│
├── notebooks/
│
├── src/
│   ├── train_model.py
│   └── sustainability_model.joblib
│
├── .gitignore
├── README.md
└── requirements.txt