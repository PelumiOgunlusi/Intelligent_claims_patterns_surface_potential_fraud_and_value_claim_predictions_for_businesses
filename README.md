# README

## Intelligent Claims Patterns: Surface Potential Fraud and Value Claim Predictions for Businesses

### Author
PELUMI OGUNLUSI

---

## Project Overview

This project aims to identify the best-performing machine learning model for detecting fraudulent insurance claims using a comprehensive dataset of insurance claim records. The workflow covers data understanding, cleaning, exploratory analysis, feature engineering, model training, evaluation, interpretation, and deployment.

---

## Dataset

- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/992mh7dk9y/2)
- **Description:** Each row represents an insurance claim with features such as customer demographics, claim details, and the target variable `fraud_reported` (Y/N).
- **Size:** 1000 rows × 40 columns
- **Privacy:** Personal identifiers are anonymized.

---

## Workflow Summary

### 1. Data Understanding & Preparation

- Loaded and inspected the dataset for structure, missing values, and duplicates.
- Visualized the distribution of the target variable (`fraud_reported`), revealing significant class imbalance.

### 2. Data Cleaning & Preprocessing

- Dropped columns with no values and rows with missing critical information.
- Replaced ambiguous values (`?`) with meaningful placeholders.
- Detected and normalized outliers in the `umbrella_limit` column.

### 3. Exploratory Data Analysis (EDA)

- Generated descriptive statistics and visualizations (histograms, boxplots, pairplots, heatmaps).
- Explored feature distributions and relationships with the target variable.

### 4. Feature Engineering

- Converted date columns to datetime and extracted year features.
- Encoded categorical variables using one-hot encoding.
- Selected top 20 features using an embedded method (ExtraTreesClassifier).
- Balanced the dataset using SMOTE oversampling.

### 5. Modeling & Evaluation

Trained and evaluated the following models:
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier (with hyperparameter tuning)
- Random Forest Classifier
- AdaBoost Classifier (with hyperparameter tuning)
- Gradient Boosting Classifier
- Extra Trees Classifier
- Voting Classifier (ensemble of above models)

**Metrics Used:**
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

### 6. Model Comparison

- Compiled and visualized model performance scores.
- Identified the best-performing model(s) for deployment.

### 7. Model Interpretation

- Used SHAP (SHapley Additive exPlanations) to interpret feature importance and model predictions.

### 8. Deployment

- Saved the trained model using `joblib`.
- Provided example code for loading the model and making predictions.
- Outlined steps to deploy the model as an API using FastAPI and Uvicorn.

---

## How to Run

1. **Install dependencies:**
    ```
    pip install pandas scikit-learn imbalanced-learn matplotlib seaborn plotly itables joblib fastapi uvicorn shap
    ```

2. **Run the notebook or Quarto document** to reproduce the analysis and modeling steps.

3. **Deploy as API:**
    - Save your trained model (already shown in the notebook).
    - Create a FastAPI app to serve predictions.
    - Run with:
      ```
      uvicorn app:app --reload
      ```

---

## Key Insights

- The dataset is highly imbalanced; SMOTE was used to address this.
- Feature selection and engineering significantly improved model performance.
- Ensemble models (Voting Classifier) and tree-based models performed best.
- SHAP analysis provided transparency into model decisions.

---

## File Structure

- `fraud_prediction.qmd` — Main Quarto notebook with code and analysis.
- `data/insurance_claims.csv` — Dataset (not included here; download from source).
- `extra_trees_model.pkl` — Example saved model.
- `README.md` — Project summary and instructions.

---

## References

- [Feature Selection Techniques in Machine Learning](https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/#h-embedded-methods)
- [Mendeley Data: Insurance Claims Dataset](https://data.mendeley.com/datasets/992mh7dk9y/2)

---

## Contact

For questions or collaboration, please contact the author via GitHub.



