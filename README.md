# CreditWise — Loan Approval Prediction

A machine learning classification project that explores applicant and financial attributes to predict whether a loan application will be approved.

> **Project focus:** Data cleaning, exploratory data analysis, categorical encoding, feature scaling, feature engineering, model comparison, and classification evaluation.

## Project Overview

Loan approval decisions involve assessing multiple applicant attributes, including income, credit score, existing liabilities, savings, collateral, and loan details. This project uses a structured dataset to build and evaluate classification models that predict the `Loan_Approved` target.

The notebook follows a practical data science workflow: inspect and prepare the data, explore relationships in the dataset, transform features, train multiple models, and compare their performance using classification metrics.

## Dataset

The notebook loads a CSV file named `loan_approval_data.csv`.

- **Rows:** 1,000
- **Original columns:** 20
- **Target:** `Loan_Approved`
- **Example input features:** Applicant income, co-applicant income, employment status, age, marital status, dependents, credit score, existing loans, debt-to-income (DTI) ratio, savings, collateral value, loan amount, loan term, loan purpose, property area, education level, gender, and employer category.

The notebook shows 950 non-null values in the initial dataset inspection for the listed fields, then applies missing-value imputation.

**Dataset note:** The CSV is an input dependency and is not embedded in the notebook. Add the dataset to the repository (only if you have permission to share it), or document where it can be obtained.

## Workflow

### 1. Data inspection and cleaning
- Loaded the dataset with Pandas.
- Inspected data types, descriptive statistics, and missing values.
- Imputed numerical columns using the mean.
- Imputed categorical columns using the most frequent value.
- Removed `Applicant_ID` because it is an identifier rather than a useful predictive feature.

### 2. Exploratory Data Analysis
Used Matplotlib and Seaborn to examine:
- Loan approval class distribution.
- Education-level counts.
- Applicant and co-applicant income distributions.
- Income, credit score, DTI ratio, and savings in relation to loan approval.
- Correlations between numerical features and the target.

The notebook's correlation analysis shows `Credit_Score` as the strongest positive numerical correlation with `Loan_Approved` among the listed features (approximately 0.45). Correlation is descriptive and does not establish causation.

### 3. Feature preparation
- Label-encoded `Education_Level` and `Loan_Approved`.
- One-hot encoded selected categorical features: employment status, marital status, loan purpose, property area, gender, and employer category.
- Split the data into training and test sets using an 80/20 split (`random_state=42`).
- Standardized features using `StandardScaler`.

### 4. Model training and comparison
The notebook evaluates three classification algorithms:
- Logistic Regression
- K-Nearest Neighbors (KNN, `n_neighbors=5`)
- Gaussian Naive Bayes

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### 5. Feature engineering
The notebook creates squared features for:
- `DTI_Ratio`
- `Credit_Score`

It then trains and evaluates the same three model types again using the transformed feature set.

## Results

The following values are taken from the notebook's displayed evaluation outputs.

### Initial feature set

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.5% | 78.3% | 77.0% | 77.7% |
| KNN | 76.0% | 62.7% | 52.5% | 57.1% |
| Gaussian Naive Bayes | 86.5% | 80.4% | 73.8% | 76.9% |

### After feature engineering

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | **88.0%** | 78.5% | **83.6%** | **81.0%** |
| KNN | 78.5% | 67.3% | 57.4% | 61.9% |
| Gaussian Naive Bayes | 86.0% | **81.1%** | 70.5% | 75.4% |

**Best accuracy and F1-score in the notebook:** Logistic Regression after feature engineering, with 88.0% accuracy and an F1-score of approximately 0.81.

Model selection depends on the business objective. For a lending workflow, false approvals and false rejections can have different costs, so precision, recall, the confusion matrix, and threshold selection should be considered—not accuracy alone.

## Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Methods:** Imputation, label encoding, one-hot encoding, standardization, train/test split, feature engineering, classification, model evaluation
- **Environment:** Jupyter Notebook

## Run Locally

1. Clone or download this repository.
2. Place `credit-wise-loanapproval.ipynb` and `loan_approval_data.csv` in the project directory.
3. Install the dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `credit-wise-loanapproval.ipynb` and run the cells in order.

## What This Project Demonstrates

- A complete, notebook-based supervised machine learning workflow.
- Data quality checks and missing-value handling.
- Exploratory analysis of applicant and financial variables.
- Transformation of categorical and numerical features.
- Comparison of multiple classification algorithms.
- Evaluation using precision, recall, F1-score, accuracy, and confusion matrices.
- An initial attempt at feature engineering and model improvement.

## Limitations and Next Improvements

This notebook is an educational / portfolio model, not a production-ready lending decision system.

A few improvements would make the experiment more robust:

1. **Prevent preprocessing leakage:** In the current notebook, imputers and encoders are fitted before the train/test split. Fit preprocessing only on training data, ideally with a Scikit-learn `Pipeline` and `ColumnTransformer`, then evaluate on an untouched test set.
2. **Use stratified splitting:** Consider `stratify=y` to preserve the target class proportions in both splits.
3. **Add cross-validation and tuning:** Compare models using stratified cross-validation and tune hyperparameters rather than relying on one split and default settings.
4. **Report class-level performance:** Include a classification report, normalized confusion matrix, and possibly ROC-AUC / PR-AUC.
5. **Review feature choices:** The feature-engineering section removes the original `Credit_Score` and `DTI_Ratio` columns after creating their squared versions. Compare this choice against retaining both original and transformed features.
6. **Assess fairness and explainability:** Review performance across relevant applicant groups and use explainability methods before considering any real lending application.
7. **Document the dataset source and license:** Include provenance, permitted use, and any relevant limitations.

## Repository Contents

```text
.
├── credit-wise-loanapproval.ipynb
├── loan_approval_data.csv   # required input; include only if sharing is permitted
└── README.md
```

## Disclaimer

This project is for learning and portfolio demonstration. Its predictions should not be used to approve or reject real loan applications. Real lending decisions require validated data, rigorous testing, regulatory and fairness review, explainability, privacy safeguards, and human oversight.
