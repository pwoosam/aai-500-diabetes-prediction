#  Diabetes Prediction Model Analysis

This project analyzes various machine learning models to predict diabetes using a structured feature selection and evaluation pipeline. It leverages statistical techniques and model interpretability for informed decision-making in healthcare analytics.

##  Project Scope

- **Dataset**: `diabetes_prediction_dataset.csv`
- **Objective**: Evaluate and compare model performance for predicting diabetes.

##  Workflow Overview

1. **Data Inspection  Cleaning**
   - Null, NA, and negative value checks
   - Feature encoding

2. **Feature Selection**
   - Techniques used: ANOVA F-value, Chi-Squared, Random Forest Importance, RFE
   - Selected top features for final modeling

3. **Model Development**
   - Models Implemented:
     - Generalized Linear Model (GLM)
     - Logistic Regression
     - K-Nearest Neighbors (KNN)
     - Decision Tree Classifier (Balanced)
     - Decision Tree Classifier (Recall Weighted)
     - Optional: Decision Tree with SMOTE
   - Each model evaluated independently and with common features

4. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC, PR AUC
   - Visuals: ROC and Precision-Recall curves
   - Summary comparison across models

##  Code Architecture

- `Model` class: Modular handling of training, prediction, and evaluation
- `FormalModel` class: Refactored version for structured model experiments
- `ModelAnalysis`: Batch evaluator for multiple models


##  Dependencies

- Python, pandas, numpy
- scikit-learn
- statsmodels
- matplotlib, seaborn

##  How to Run

```bash
git clone <repo_url>
cd <repo_folder>
pip install -r requirements.txt
jupyter notebook ModelAnalysis.ipynb
```

##  Dual Flow Architecture Explained

This notebook is structured around **two complementary model analysis flows**, each serving a different analytical purpose:

---

###  Flow 1: **Individual Model Exploration with Manually Selected Features**

####  Objective:
To allow **independent experimentation** of different models using **customized or domain-driven feature sets** without enforcing consistency across models.

####  What Happens:
- Each model is fed **different combinations of features**, chosen based on intuition, prior domain knowledge, or exploratory analysis.
- SMOTE is selectively applied to test performance on imbalanced datasets.
- Evaluation includes:
  - Classification metrics (Accuracy, Precision, Recall, F1)
  - ROC and PR curves
  - Confusion matrix

####  Benefits:
- Helps understand **how sensitive each model is to different feature sets**.
- Encourages **model-specific optimization** (e.g., tuning Decision Tree on a different feature mix than Logistic Regression).
- Useful in a research/experimentation phase to see what works before standardizing.

####  Limitations:
- Results are **not directly comparable** across models due to inconsistent inputs.
- Can lead to misleading conclusions if used for performance benchmarking.

---

###  Flow 2: **Feature-Selected Pipeline with Structured Model Benchmarking**

####  Objective:
To enable a **fair, consistent, and data-driven comparison** of all models using a common feature set derived from formal feature selection.

####  Process:
1. **Statistical Feature Selection**:
   - ANOVA F-test
   - Chi-squared test
   - Random Forest Importance
   - Recursive Feature Elimination (RFE)
2. **Top features are selected** programmatically based on aggregate scoring.
3. **Same set of features** is passed to all models:
   - Logistic Regression
   - KNN
   - Decision Tree (Balanced / Weighted)
   - GLM

4. Metrics such as Accuracy, F1, ROC AUC, PR AUC are tabulated for **side-by-side model comparison**.

####  Benefits:
- Enables **apples-to-apples comparison** of models.
- Ensures that improvements are due to modeling technique, not input features.
- More appropriate for final model selection and reporting.

####  Example Comparison Table:
| Model                        | F1 Score | Precision | Recall | ROC AUC | PR AUC |
|-----------------------------|----------|-----------|--------|---------|--------|
| Logistic Regression         | 0.73     | 0.86      | 0.64   | 0.96    | 0.82   |
| Decision Tree (Recall Wt)   | 0.71     | 0.70      | 0.74   | 0.85    | 0.58   |
| Generalized Linear Model    | 0.10     | 0.45      | 0.06   | 0.59    | 0.18   |

---

##  Summary

| Flow            | Purpose                             | Features Used         | Outcome                           |
|-----------------|--------------------------------------|------------------------|------------------------------------|
| **Flow 1**      | Per-model tuning and exploration     | Varies by model        | Exploratory insights, local bests  |
| **Flow 2**      | Structured, reproducible comparison  | Common selected set    | Reliable benchmark for reporting   |
