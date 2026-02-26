# 🎓 Student Dropout Prediction — Machine Learning Final Project

A complete end-to-end binary classification pipeline to predict **student dropout risk** using behavioral and engagement data. This project covers data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and model interpretation with SHAP and LIME.

> **Team Members:**
> - Pham Gia Khiem — 551026 (Vietnam)
> - Mohammed Hassan — 541140 (Bangladesh)
> - Fazlur Rahman — 541927 (Bangladesh)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Montasir00/Ml_final_project/blob/tabby/19-5.ipynb)

---

## 📖 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [I. Understanding the Dataset](#i-understanding-the-dataset)
  - [II. Data Preprocessing](#ii-data-preprocessing)
  - [III. Exploratory Data Analysis](#iii-exploratory-data-analysis-eda)
  - [IV. Feature Engineering](#iv-feature-engineering)
  - [V. Modeling](#v-modeling)
  - [VI. Hyperparameter Tuning](#vi-hyperparameter-tuning)
  - [VII. Model Interpretation](#vii-model-interpretation)
- [Results Summary](#results-summary)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)

---

## 🔍 Project Overview

This project builds machine learning models to **classify whether a student is at risk of dropping out** based on features related to their engagement, behavior, and demographics. The task is a **binary classification** problem with the target variable `Dropout Status` (0 = Not dropped out, 1 = Dropped out).

The full pipeline:

```
Raw Data → EDA → Preprocessing → Feature Engineering
→ Scaling → Model Training → Hyperparameter Tuning
→ SHAP & LIME Interpretation → Final Insights
```

---

## 📦 Dataset

| File | Description |
|---|---|
| `final_project_dataset_complete.csv` | Raw dataset with numerical and categorical features |
| `cleaned_dataset.csv` | After missing value handling, outlier treatment & encoding |
| `cleaned_dataset with new features.csv` | After feature engineering |

**Key Features (after renaming for interpretability):**

| Original Name | Renamed | Description |
|---|---|---|
| `feature_2` | Course Engagement Score (0–10) | How actively engaged the student is |
| `feature_4` | Avg Time Spent Per Session (min) | Average session duration |
| `feature_9` | Number of Missed Deadlines | Engineered: `feature_2 + category_1_encoded` |
| `category_1_encoded` | Student Type | Label-encoded student category |
| `target` | Dropout Status | Binary: 0 = active, 1 = dropout |

**Categorical Features:**
- `category_1` — Student type (label encoded)
- `category_2` — Regional grouping (one-hot encoded → Region A, Region B, Region C)

---

## 📁 Project Structure

```
student-dropout-prediction/
├── Final.ipynb                               # Main analysis notebook
├── final_project_dataset_complete.csv        # Raw dataset
├── cleaned_dataset.csv                       # Preprocessed data
├── cleaned_dataset with new features.csv     # Feature-engineered data
├── metrics_comparison.png                    # Default model metrics chart
├── roc_comparison.png                        # Default model ROC curves
├── cv_scores_comparison.png                  # Cross-validation boxplots
├── tuned_metrics_comparison.png              # Tuned model metrics chart
├── tuned_roc_comparison.png                  # Tuned model ROC curves
├── feature_importance_comparison.png         # RF vs GB feature importance
├── default_vs_tuned_comparison.png           # Default vs Tuned comparison
└── README.md
```

---

## 🔄 Pipeline Walkthrough

### I. Understanding the Dataset

```python
df = pd.read_csv('final_project_dataset_complete.csv')
df.info()
df.describe()
```

Initial analysis:
- Identified **numerical** vs **categorical** features
- Computed the **correlation matrix** for numerical features
- Visualized feature distributions with **histograms** and **boxplots**

---

### II. Data Preprocessing

#### 1. Duplicate & Missing Value Check

```python
df.duplicated().sum()   # No duplicate rows
df.isna().sum()         # Identified missing values in feature_3 and feature_6
```

Missing values found in `feature_3` and `feature_6` — both imputed with **median** (robust to skew):

```python
imputer = SimpleImputer(strategy='median')
df['feature_3'] = imputer.fit_transform(df[['feature_3']])
df['feature_6'] = imputer.fit_transform(df[['feature_6']])
```

#### 2. Outlier Detection & Treatment

Used the **IQR method** (1.5× multiplier) across all numeric columns:

```python
def iqr_outliers(dataset, feature_name, multiplier=1.5):
    Q1 = dataset[feature_name].quantile(0.25)
    Q3 = dataset[feature_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - multiplier * IQR
    upper_limit = Q3 + multiplier * IQR
    outliers = dataset[(dataset[feature_name] < lower_limit) | (dataset[feature_name] > upper_limit)]
    return outliers, {'lower_limit': lower_limit, 'upper_limit': upper_limit}
```

Outliers were **replaced with the median** of each feature (rather than dropped) to preserve sample size.

#### 3. Categorical Encoding

```python
# Label Encoding for category_1 (ordinal)
label_encoder = LabelEncoder()
df['category_1_encoded'] = label_encoder.fit_transform(df['category_1'])

# One-Hot Encoding for category_2 (nominal)
df_one_hot = pd.get_dummies(df['category_2']).astype(int)
df = pd.concat([df, df_one_hot], axis=1)
df.drop(columns=['category_1', 'category_2'], inplace=True)
```

✅ Cleaned dataset exported to `cleaned_dataset.csv`

---

### III. Exploratory Data Analysis (EDA)

#### Statistical Tests

**T-tests** (numerical features vs target):
```python
for feature in numerical_features:
    group1 = df[df['target'] == 1][feature].dropna()
    group0 = df[df['target'] == 0][feature].dropna()
    t_stat, p_value = stats.ttest_ind(group1, group0)
```
Identified which numerical features differ significantly between dropout and non-dropout groups.

**Chi-square tests** (categorical features vs target):
```python
# Tested: category_1_encoded, Region A, Region B, Region C
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
```
Determined statistical association between regional/student-type categories and dropout.

#### Visualizations
- Correlation heatmap of all features
- Histograms of feature distributions
- Boxplots before/after outlier treatment
- Scatter plots and pairplots of top correlated features with target

---

### IV. Feature Engineering

One new feature was created and several low-signal features were dropped:

```python
def feature_engineering(df):
    # New interaction feature
    df['feature_9'] = df['feature_2'] + df['category_1_encoded']
    # → Later renamed: "Number of Missed Deadlines"

    # Drop weak or redundant features
    df = df.drop([
        'feature_1', 'feature_3', 'feature_5',
        'feature_6', 'feature_7', 'feature_8',
        'Region A', 'Region B', 'Region C'
    ], axis=1)
    return df
```

Final feature set retained: `feature_2`, `feature_4`, `feature_9`, `category_1_encoded`, `target`

#### Feature Scaling

All features normalized using **MinMaxScaler** (scales to [0, 1]):

```python
scaler = MinMaxScaler()
for feature in df_3:
    df_4[feature] = scaler.fit_transform(df_4[[feature]])
```

✅ Engineered dataset exported to `cleaned_dataset with new features.csv`

---

### V. Modeling

**Train/Test Split:** 80% train, 20% test (`random_state=42`)

```python
X = df_4.drop('target', axis=1)
y = df_4['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Three models were trained and evaluated:

#### Model 1: Random Forest (Default)

```python
rf_default = RandomForestClassifier(random_state=42)
```

Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC, 5-Fold Stratified Cross-Validation

#### Model 2: Gradient Boosting (Default)

```python
gb_default = GradientBoostingClassifier()
```

Same full evaluation suite as Random Forest.

#### Model 3: Soft Voting Classifier (Default)

Ensemble of three base learners with **soft voting** (probability averaging):

```python
clf1 = LogisticRegression(random_state=42)
clf2 = SVC(probability=True, random_state=42)
clf3 = DecisionTreeClassifier(random_state=42)

soft_voting = VotingClassifier(
    estimators=[('lr', clf1), ('svc', clf2), ('dt', clf3)],
    voting='soft'
)
```

---

### VI. Hyperparameter Tuning

#### Random Forest — GridSearchCV

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_leaf_nodes': [5, 10, 15]
}
grid_search = GridSearchCV(rf_tuned, param_grid, cv=5)
```

**Best parameters found:**
```
n_estimators=300, max_depth=10, max_leaf_nodes=15,
min_samples_leaf=1, min_samples_split=5
```

#### Gradient Boosting — RandomizedSearchCV

```python
gb_param = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6],
    'subsample': [0.6, 0.8, 1.0],
    'max_features': ['sqrt', 'log2', None]
}
gb_random_search = RandomizedSearchCV(gb_tuning, gb_param, cv=5, n_iter=30, random_state=42)
```

**Best parameters found:**
```
n_estimators=200, max_depth=5, learning_rate=0.01,
min_samples_split=2, min_samples_leaf=1
```

#### Voting Classifier — RandomizedSearchCV

```python
vt_params = {
    'lr__C': [0.1, 1, 10],
    'svc__C': [0.1, 1, 10],
    'dt__max_depth': [5, 10, 15],
    'dt__min_samples_split': [2, 5, 10],
    'dt__min_samples_leaf': [1, 2, 4]
}
```

**Best tuned components:**
```
LogisticRegression(C=1), SVC(C=0.1), DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)
```

---

### VII. Model Interpretation

#### SHAP (SHapley Additive exPlanations)

Applied to both the tuned **Random Forest** and **Gradient Boosting** models:

```python
explainer = shap.TreeExplainer(rf_tuning)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0, :, 0])   # Individual prediction explanation
shap.summary_plot(shap_values[:, :, 0])       # Global feature importance (Class 0)
shap.summary_plot(shap_values[:, :, 1])       # Global feature importance (Class 1)
```

SHAP summary plots generated for both classes (dropout / not dropout), revealing the direction and magnitude of each feature's influence.

#### LIME (Local Interpretable Model-agnostic Explanations)

Applied to explain individual test set predictions using the tuned Voting Classifier:

```python
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)
exp = explainer.explain_instance(sample.values, predict_fn=soft_voting_tuned.predict_proba)
exp.show_in_notebook(show_table=True)
```

#### Top Drivers of Dropout Risk

Based on model interpretation (SHAP + feature importance), the top predictors of student dropout are:

| Rank | Feature | Avg Impact |
|---|---|---|
| 🥇 1 | **Course Engagement Score** | 0.65 |
| 🥈 2 | **Number of Missed Deadlines** | 0.40 |
| 🥉 3 | **Avg Time Spent Per Session** | 0.25 |
| 4 | **Student Type** | 0.10 |

> Students with **low engagement scores** and **high missed deadlines** are most at risk of dropping out.

---

## 📊 Results Summary

### Default Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Random Forest | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — |
| Voting Classifier | — | — | — | — | — |

### Tuned Models

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Random Forest (Tuned) | — | — | — | — | — |
| Gradient Boosting (Tuned) | — | — | — | — | — |
| Voting Classifier (Tuned) | — | — | — | — | — |

> ℹ️ Actual metric values are available in the notebook outputs. The tuned models are compared against their default counterparts with Default vs Tuned accuracy and AUC improvement plots.

All models were evaluated with **5-fold Stratified Cross-Validation** to ensure reliable, unbiased performance estimates.

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and DataFrames |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization (histograms, heatmaps, boxplots, ROC curves) |
| `scipy` | T-tests and Chi-square statistical tests |
| `scikit-learn` | Full ML pipeline: preprocessing, models, tuning, evaluation |
| `shap` | Global and local model interpretation (TreeExplainer) |
| `lime` | Local prediction explanation (LimeTabularExplainer) |

**Models used:**
- `RandomForestClassifier` — Ensemble of decision trees
- `GradientBoostingClassifier` — Sequential boosting ensemble
- `VotingClassifier` (soft) — Ensemble of Logistic Regression, SVC, Decision Tree
- `GridSearchCV` — Exhaustive hyperparameter search (Random Forest)
- `RandomizedSearchCV` — Randomized hyperparameter search (Gradient Boosting, Voting)

---

## ⚙️ Getting Started

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy shap lime jupyter
```

### Run the Notebook

```bash
git clone https://github.com/Montasir00/Ml_final_project.git
cd Ml_final_project
jupyter notebook Final.ipynb
```

Or open directly in Google Colab using the badge at the top of this README.

### Required Data File

Place `final_project_dataset_complete.csv` in the same directory as the notebook before running.

### Notebook Execution Order

Run all cells in order — the notebook is structured in 7 sequential sections:

```
I.   Understanding the Dataset
II.  Data Preprocessing
III. Exploratory Data Analysis
IV.  Feature Engineering
V.   Modeling
VI.  Hyperparameter Tuning
VII. Model Interpretation (SHAP + LIME)
```
