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
4. **Access the API:**
    - Open your browser and navigate to `https://intelligent-claims-patterns-surface-gszs.onrender.com/docs` to interact with the API.

5. Note: When running the API, ensure you put in the FraudFeatures variables 
  in the correct format as shown in the `fraud_cases.py` file. Majority of the features in this class are boolean features, taking only two values: 0 for No and 1 for Yes. The only features that take integer values (other than 0 or 1) are:
- bodily_injuries
- insured_zip
- total_claim_amount
- policy_number
- vehicle_claim
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
## For live interactive input and prediction, visit this link:
[Interactive Input](https://intelligent-claims-patterns-surface-gszs.onrender.com/docs) 

For questions or collaboration, please contact the author via GitHub.

---

## Replicating This Project

To replicate the entire workflow and achieve similar results:

1. **Clone the Repository**
    ```bash
    git clone https://github.com/PelumiOgunlusi/Intelligent_claims_patterns_surface_potential_fraud_and_value_claim_predictions_for_businesses
    cd Intelligent_claims_patterns_surface_potential_fraud_and_value_claim_predictions_for_businesses
    ```

2. **Download the Dataset**
    - Visit [Mendeley Data](https://data.mendeley.com/datasets/992mh7dk9y/2) and download `insurance_claims.csv`.
    - Place the file in the `data/` directory.

3. **Set Up the Environment**
    - Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
      Or use the list provided in the "How to Run" section.

4. **Run the Analysis**
    - Open and execute `fraud_prediction.qmd` in Quarto or Jupyter Notebook to reproduce all steps: data loading, cleaning, EDA, feature engineering, modeling, and evaluation.

5. **Model Deployment (Optional)**
    - Follow the deployment instructions to serve the trained model as an API using FastAPI and Uvicorn.

6. **Verify Results**
    - Compare your model metrics and SHAP visualizations with those reported in this README to ensure consistency.

**Note:** For best reproducibility, use the same Python version and package versions as specified in the repository.


