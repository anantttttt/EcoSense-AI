# EcoSense AI Architecture

## System Workflow

```text
                    ┌──────────────────────┐
                    │        USER          │
                    │ Sustainability Data  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  STREAMLIT INTERFACE  │
                    │                      │
                    │ • Electricity        │
                    │ • Transportation     │
                    │ • Waste              │
                    │ • Water              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   INPUT PROCESSING   │
                    │                      │
                    │ Feature preparation  │
                    │ Transport encoding   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   ML CLASSIFIER      │
                    │                      │
                    │ Decision Tree        │
                    │ Scikit-learn         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ AI CLASSIFICATION    │
                    │                      │
                    │ • Excellent          │
                    │ • Good               │
                    │ • Moderate            │
                    │ • Needs Improvement   │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
       │  DECISION    │ │    IMPACT    │ │ SUSTAINABILITY│
       │   FACTORS    │ │   ANALYSIS   │ │   DASHBOARD  │
       └──────┬───────┘ └──────┬───────┘ └──────────────┘
              │                │
              └────────┬───────┘
                       ▼
              ┌──────────────────┐
              │ RECOMMENDATIONS  │
              │                  │
              │ Practical steps  │
              │ for improvement  │
              └──────────────────┘
## Model Training Workflow

```text
Synthetic Dataset
       │
       ▼
1,000 Sustainability Profiles
       │
       ▼
Train / Test Split
       │
       ▼
Decision Tree Classifier
       │
       ▼
Model Evaluation
       │
       ▼
sustainability_model.joblib
       │
       ▼
EcoSense AI Application

### What each box means

**1. Synthetic Dataset**

We generated our training data specifically for the prototype.

↓

**2. 1,000 Sustainability Profiles**

Our dataset contains 1,000 generated examples involving:

- Electricity
- Transportation
- Waste
- Water
- Sustainability category

↓

**3. Train / Test Split**

We split the data into:

- **80% training**
- **20% testing**

So approximately:

```text
800 → training
200 → testing

4. Decision Tree Classifier

Our actual ML algorithm.

It learns patterns from the training data and predicts one of:

Excellent
Good
Moderate
Needs Improvement
5. Model Evaluation

We test the trained model against the 20% it didn't train on and calculate things such as accuracy and the classification report.

↓

6. sustainability_model.joblib

This is the trained model saved to:

src/sustainability_model.joblib

That's the file your Streamlit application loads.

↓

7. EcoSense AI Application

When someone enters their electricity, transportation, waste and water values, app.py loads that saved model and asks it for a prediction.