
# Titanic Survival Prediction Pipeline

Predicting survival on the Titanic using structured machine learning and principled model evaluation.

---

##  Project Overview

This project focuses on building a **clean, well-evaluated machine learning pipeline** for the Kaggle *Titanic: Machine Learning from Disaster* competition.  
Rather than leaderboard chasing, the emphasis is on:

- Sound preprocessing
- Feature engineering with intuition and domain knowledge
- Proper validation discipline
- Responsible ensembling
- Understanding model behavior and limitations

---

##  Repository Structure

```
.
├── feature_creation_titanic_survival.py
├── titanic_survival_prediction_iter(1).py
├── titanic_survival_prediction_iter(2).py
├── titanic_survival_prediction_iter(3).py
├── titanic_survival_prediction_iter(4).py
├── titanic_survival_prediction_iter(5).py
├── Submission1_iter(1).csv
├── Submission1_iter(2).csv
├── Submission1_iter(3).csv
├── Submission1_iter(4).csv
├── Submission1_iter(5).csv
└── README.md
```

Each iteration represents a **conceptual improvement**, not random tuning.

---

##  Methodology

### 1. Data Preprocessing & Feature Engineering
- Handled missing values using median/mode strategies
- Engineered meaningful features:
  - `Title` (extracted from Name)
  - `Family_size`
  - `Travelling_alone`
- Separate preprocessing pipelines for:
  - Logistic Regression
  - Random Forest / XGBoost
- One-hot encoding for categorical variables
- Scaling applied only where appropriate

---

### 2. Model Iterations

- **Iter 1–2:** Baseline Logistic Regression & Random Forest
- **Iter 3–4:** Hyperparameter tuning and XGBoost introduction
- **Iter 5:** Final weighted probability ensemble

Models used:
- Logistic Regression
- Random Forest
- XGBoost

---

### 3. Evaluation Strategy

- Stratified train/validation split
- Primary metric: **F1-score**
- Secondary metrics: Accuracy, ROC-AUC
- Ensemble weights derived from **validation performance**
- All model decisions frozen before final evaluation

This avoids leaderboard overfitting and preserves generalization integrity.

---

##  Results

| Model | Validation F1 | Validation/CV Accuracy
|------|---------------|--------------------|
| Logistic Regression | ~0.738 | 81.1
| Random Forest | ~0.732 | 83.5
| XGBoost | ~0.701 | 82.8

The final ensemble slightly outperformed the strongest individual model.

A ~3.6% accuracy improvement resulted in a leaderboard jump from ~12,000 to ~2,000 out of ~14,000 participants.

---

##  Key Learnings

- Small datasets can produce misleading validation confidence
- Feature engineering often matters more than model complexity
- Ensembling works best when models have complementary inductive biases
- Knowing **when to stop experimenting** is a critical ML skill

---


##  Future Work

- Feature stability analysis
- Neural networks for learning-based representations

---

