# 🏦 Loan Approval ML Model

A comprehensive machine learning project that predicts **loan approval status** (approved/rejected) and estimates the **maximum allowable loan amount** for applicants — built across three progressive Jupyter notebooks using a real-world dataset of 58,645 records.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [Notebook 1 — EDA & Preprocessing](#notebook-1--eda--preprocessing)
  - [Notebook 2 — Classification Models](#notebook-2--classification-models)
  - [Notebook 3 — Ensemble & Regression Models](#notebook-3--ensemble--regression-models)
- [Models Used](#models-used)
- [Results Summary](#results-summary)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Project Overview

Financial institutions need fast, data-driven decisions on loan applications. This project builds and evaluates machine learning models to automate two key tasks:

1. **Binary Classification** — Predict whether a loan application will be *Approved* (1) or *Rejected* (0).
2. **Regression** — Predict the *maximum loan amount* a client is eligible for.

The project follows a full data science pipeline: raw data ingestion, exploratory analysis, preprocessing, model training, hyperparameter tuning, and performance evaluation.

---

## Repository Structure

```
loan-approval-ml-model/
└── New folder/
    ├── NootBook1.ipynb                          # EDA, preprocessing & feature engineering
    ├── NoteBook_2.ipynb                         # Classification models (LR, NB, KNN)
    ├── notebook3_WORKING (1).ipynb              # Ensemble classifier + regression models
    ├── loan_approval_data (1).csv               # Dataset (58,645 records × 13 columns)
    └── Analysis Report Template 15032026(v1.0).docx  # Project analysis report
```

---

## Dataset

**File:** `loan_approval_data (1).csv`  
**Records:** 58,645 rows × 13 columns

| Column | Type | Description |
|---|---|---|
| `id` | int | Unique applicant identifier |
| `age` | float | Applicant age (years) |
| `income` | int | Annual income |
| `home_ownership` | str | OWN / RENT / MORTGAGE / OTHER |
| `emplyment_length` | int | Years of employment |
| `loan_intent` | str | EDUCATION / MEDICAL / HOMEIMPROVEMENT / VENTURE / DEBTCONSOLIDATION / PERSONAL |
| `loan_amount` | int | Requested loan amount |
| `loan_interest_rate` | float | Interest rate (%) |
| `loan_income_ratio` | float | Loan-to-income ratio |
| `payment_default_on_file` | str | Prior default history (Y / N) |
| `credit_history_length` | int | Length of credit history (years) |
| `loan_approval_status` | int | **Target** — 0 = Rejected, 1 = Approved |
| `max_allowed_loan` | int | **Regression target** — Max eligible loan amount |

**Class distribution:**
- Rejected (0): 50,295 records (85.8%)
- Approved (1): 8,350 records (14.2%)

**Missing values:**
- `age`: 6 missing → imputed with median
- `loan_interest_rate`: 11 missing → imputed with median
- `payment_default_on_file`: 5 missing → imputed with mode

---

## Project Workflow

### Notebook 1 — EDA & Preprocessing

The first notebook focuses on understanding and cleaning the raw data.

**Exploratory Data Analysis:**
- Dataset shape and data types inspection (`df.info()`, `df.describe()`)
- Target class distribution visualised with Plotly bar charts
- Distribution of all numerical features via faceted histograms
- Outlier detection using the IQR method across all numeric columns
- Box plots for visual outlier inspection
- Removal of biologically implausible records (age > 100)

**Preprocessing steps:**
- Numeric columns (age, loan_interest_rate): missing values imputed with **mean** via `SimpleImputer`
- Categorical columns (payment_default_on_file): missing values imputed with **most frequent** value
- `id` column dropped (not a predictive feature)
- `payment_default_on_file` label-encoded (N → 0, Y → 1)
- `home_ownership` and `loan_intent` one-hot encoded with `pd.get_dummies`

---

### Notebook 2 — Classification Models

Three individual classifiers are trained and compared.

**Preprocessing (model-specific):**
- `id` and `max_allowed_loan` dropped from feature set
- All categorical columns label-encoded with `LabelEncoder`
- Remaining nulls filled with column median
- Features standardised with `StandardScaler`
- 80/20 train/test split with `stratify=y` to preserve class ratios

**Models trained:**

| Model | Notes |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Gaussian Naïve Bayes | No hyperparameter tuning required |
| K-Nearest Neighbours | Tested with k=5; optimised via GridSearchCV (k ∈ {3, 5, 7, 11}) |

**Evaluation:**
- Accuracy score and ROC-AUC for each model
- Confusion matrix visualised with Plotly heatmaps
- Full classification report (precision, recall, F1-score) for each model

---

### Notebook 3 — Ensemble & Regression Models

The final notebook adds an ensemble classifier and a full regression pipeline.

**Part A — Ensemble Classification (Logistic Regression + Naïve Bayes):**
- `VotingClassifier` with `voting='soft'` (probability-based voting)
- Compared against individual base learners on accuracy
- ROC curves and AUC plotted for all three models (LR, NB, Ensemble)
- Bar chart summary of classification accuracy across all three

**Part B — Regression (Predicting Maximum Loan Amount):**
- Target: `max_allowed_loan`; records with corrupted negative values removed
- `loan_approval_status` dropped; all categorical columns label-encoded
- 80/20 train/test split

Four regression models are evaluated:

| Model | Key Hyperparameters |
|---|---|
| Decision Tree (DT-1) | Fully grown (no depth limit) |
| Decision Tree (DT-2) | Pruned — `max_depth=5` |
| Random Forest | `n_estimators=100`, `max_depth=10` |
| Gradient Boosting | `n_estimators=100`, `max_depth=4` |

**Regression metrics:** MAE, MSE, RMSE, R²

The pruned DT-2 tree is visualised (top 3 levels) for interpretability.

**Applied prediction:** Client 60256's maximum loan amount is predicted using DT-2 based on their specific profile (age 56, income £57,000, RENT ownership, MEDICAL intent).

---

## Models Used

| Category | Model |
|---|---|
| Classification | Logistic Regression |
| Classification | Gaussian Naïve Bayes |
| Classification | K-Nearest Neighbours (KNN) |
| Classification | Voting Ensemble (LR + NB, soft voting) |
| Regression | Decision Tree Regressor (fully grown) |
| Regression | Decision Tree Regressor (pruned, depth=5) |
| Regression | Random Forest Regressor |
| Regression | Gradient Boosting Regressor |

---

## Results Summary

### Classification (loan_approval_status)

Models are evaluated on accuracy and ROC-AUC on a held-out 20% test set.

| Model | Metric |
|---|---|
| Logistic Regression | Accuracy + ROC-AUC reported |
| Naïve Bayes | Accuracy + ROC-AUC reported |
| KNN (k=5, tuned) | Accuracy + ROC-AUC reported |
| **Ensemble (LR + NB)** | **Best or comparable accuracy across all three** |

> Note: Actual numeric scores depend on the runtime environment. The ensemble with soft voting generally matches or outperforms individual base learners.

### Regression (max_allowed_loan)

| Model | Notes |
|---|---|
| DT-1 (Fully Grown) | Prone to overfitting; high variance |
| DT-2 (Pruned, depth=5) | Better generalisation; interpretable |
| Random Forest | Strong ensemble performance |
| Gradient Boosting | Competitive with Random Forest |

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or Google Colab

### Clone the repository

```bash
git clone https://github.com/sahan200401/loan-approval-ml-model.git
cd loan-approval-ml-model
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn plotly matplotlib seaborn
```

### Run the notebooks

Open the notebooks in order in Jupyter or upload to Google Colab:

1. `NootBook1.ipynb` — EDA & Preprocessing
2. `NoteBook_2.ipynb` — Classification
3. `notebook3_WORKING (1).ipynb` — Ensemble & Regression

> **Note:** Update the CSV file path in each notebook to match your local path. The notebooks currently reference `/content/loan_approval_data (1).csv` (Google Colab path).

---

## Usage

To make a prediction for a new client using the trained DT-2 regression model:

```python
import pandas as pd

client_data = {
    'age': 56.0,
    'income': 57000,
    'home_ownership': 3,           # RENT
    'emplyment_length': 15,
    'loan_intent': 3,              # MEDICAL
    'loan_amount': 25700,
    'loan_interest_rate': 23.0,
    'loan_income_ratio': 0.10,
    'payment_default_on_file': 0,  # N (No)
    'credit_history_length': 35
}

client_df = pd.DataFrame([client_data])
predicted_max_loan = dt2.predict(client_df)[0]
print(f'Predicted Maximum Loan Amount: £{predicted_max_loan:,.2f}')
```

**Encoding reference:**

| Feature | Encoding |
|---|---|
| home_ownership | MORTGAGE=0, OTHER=1, OWN=2, RENT=3 |
| loan_intent | DEBTCONSOLIDATION=0, EDUCATION=1, HOMEIMPROVEMENT=2, MEDICAL=3, PERSONAL=4, VENTURE=5 |
| payment_default_on_file | N=0, Y=1 |

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `plotly` | Interactive charts and visualisations |
| `matplotlib` | Static plots (decision tree visualisation) |
| `seaborn` | Statistical visualisations |
