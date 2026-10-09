# Tabular ML Notebook Cheat Sheet (Classification)

**Workflow:** Load → clean labels and documented missing markers → split → explore training data → select columns by dtype → build pipeline → cross-validate → fit → evaluate once.

The **UCI Adult Income** dataset is the main runnable example. No feature-name lists are hardcoded: numeric and categorical inputs are selected from pandas dtypes. **Important exception:** dtype is only a default heuristic; numeric-coded categories, IDs, dates and target-derived columns may require explicit overrides after inspection.

> **Rule:** Learn all fitted preprocessing inside the cross-validation pipeline. Don't choose hyperparameters, thresholds, or features using final test scores.

## 1. Complete UCI Adult example — copy into a notebook

Install once in your chosen Python environment: `python -m pip install ucimlrepo pandas numpy scikit-learn matplotlib ipython`.

```python
# 0. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

# 1. Load; features and target are separate in ucimlrepo
adult = fetch_ucirepo(id=2)
X = adult.data.features.copy()
y = adult.data.targets.squeeze().copy()  # pandas Series (not DataFrame)
X.columns = X.columns.str.strip()

# 2. Deterministic text cleanup (does NOT learn any statistics)
# Inspect your dataset's documentation before treating tokens as missing.
# Adult may include '?' or padded '?' in categorical inputs.
text_cols = X.select_dtypes(include=["object", "string", "category"]).columns
for col in text_cols:
    X[col] = X[col].astype("string").str.strip().replace({"?": np.nan, "": np.nan})
    # Convert to plain object so sklearn SimpleImputer consistently sees np.nan.
    X[col] = X[col].astype(object).where(X[col].notna(), np.nan)

# Adult income labels occur with and without a final dot; map to binary:
# 0 = <=50K, 1 = >50K
print("Raw target labels:")
print(y.value_counts(dropna=False))
y = y.astype("string").str.strip().str.rstrip(".").map({"<=50K": 0, ">50K": 1})
assert y.notna().all(), "Unexpected target value: inspect y before proceeding"
y = y.astype(int)

# 3. Train/test split BEFORE fitted preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Explore TRAINING data (more checks in section 2)
print("Shape:", X_train.shape)
display(X_train.head())
X_train.info()
display(pd.DataFrame({
    "missing": X_train.isna().sum(),
    "missing_%": (100 * X_train.isna().mean()).round(2),
    "unique": X_train.nunique(dropna=True),
}).sort_values("missing", ascending=False))
display(X_train.describe(include="number").T)
print("Target counts:")
display(y_train.value_counts())
display(y_train.value_counts(normalize=True))

# 5. Select automatically by DTYPE, not hardcoded names
numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = X_train.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

# Sanity check: no feature silently dropped (e.g. datetime / extension dtype)
selected = set(numeric_cols) | set(categorical_cols)
assert selected == set(X_train.columns), (
    f"Unassigned columns: {set(X_train.columns) - selected}. "
    "Inspect and explicitly convert or drop these features."
)

# OPTIONAL semantic overrides AFTER inspection:
# A numeric-coded nominal category should be one-hot encoded, not scaled.
# categorical_cols += ["some_numeric_category"]
# numeric_cols.remove("some_numeric_category")
# Similarly, explicitly REMOVE IDs/leaky predictors from X before splitting.

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # remove if none missing
    ("scaler", StandardScaler()),                      # optional for tree models
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    # Alternative: SimpleImputer(strategy="most_frequent")
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

# 6. Cross-validation of COMPLETE pipeline on training partition only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
print("F1 per fold:", scores)
print("Mean CV F1:", scores.mean())

# 7. Train original pipeline and evaluate untouched test set
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Test F1 (>50K):", f1_score(y_test, y_pred))
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(classification_report(
    y_test, y_pred, labels=[0, 1], target_names=["<=50K", ">50K"],
    zero_division=0,
))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, labels=[0, 1],
    display_labels=["<=50K", ">50K"], cmap="Blues", values_format="d",
)
plt.show()
```

## 2. Thorough data exploration checklist

Start with `X_train` and `y_train` **after splitting**. Some items need interpretation, not automatic cleaning.

### Shape, types, unique values, and candidate identifiers

```python
print("Shape:", X_train.shape)
X_train.info()
display(X_train.head(10))
print("Unique values per feature:")
display(X_train.nunique(dropna=False).sort_values())
print("Column names:", X_train.columns.tolist())
```

- **Column name whitespace:** `raw.columns = raw.columns.str.strip()`.
- **Unexpected type:** numeric-looking strings may need `pd.to_numeric(..., errors="coerce")`; category codes (e.g. `pclass`) may need categorical encoding, **not** numeric scaling.
- **IDs / leakage:** exclude record identifiers, target-derived features, or columns unavailable when making a real prediction.
- **High cardinality:** check columns with very many categories (possible free text, IDs, or one-hot explosion).

### Missing values: true nulls **and disguised nulls**

`isna()` catches real `NaN`, `None`, `pd.NA`, and `NaT`; it **does not** catch strings such as `"?"`, `"NA"`, `"null"`, or whitespace-only entries.

```python
# A. Genuine missingness
summary = pd.DataFrame({
    "missing": X_train.isna().sum(),
    "missing_%": 100 * X_train.isna().mean(),
    "unique": X_train.nunique(dropna=True),
}).sort_values("missing", ascending=False)
display(summary.round(2))

# B. Look at most common values of every categorical column
for col in X_train.select_dtypes(include=["object", "category", "string"]).columns:
    print(f"\n{col}")
    display(X_train[col].value_counts(dropna=False).head(15))

# C. Audit suspicious literal tokens, ignoring surrounding whitespace/case
suspect_tokens = {"", "?", "na", "n/a", "null", "none", "nan", "missing", "-"}
for col in X_train.select_dtypes(include=["object", "category", "string"]).columns:
    s = X_train[col].astype("string").str.strip().str.lower()
    counts = s[s.isin(suspect_tokens)].value_counts(dropna=False)
    if not counts.empty:
        print(f"Possible placeholders in {col}:")
        print(counts.to_string())
```

**Important:** A token like `"None"`, `"Unknown"`, `"-"`, or `"NA"` might be a **valid category** (e.g. “none of the above,” or NA as a code). Confirm what it means in the dataset documentation **before** replacing it. Case-insensitive inspection is safer than blind replacement.

If your dataset uses padded markers such as `" ? "`, standardise them in the **load/clean** step before splitting:

```python
# OPTIONAL: for datasets with string features containing padded labels or '?' markers
# Run on selected X before train_test_split. Deterministic cleanup, not learned fitting.
obj_cols = X.select_dtypes(include=["object", "string"]).columns
for col in obj_cols:
    X[col] = X[col].str.strip()
X = X.replace({"?": np.nan, "": np.nan})  # add ONLY documented null tokens
```

If you use this optional block, rerun the split afterwards. The pipeline imputes remaining missing values **within CV folds**, avoiding leakage.

**Overlap of missingness:**

```python
# Number of rows missing ANY value (not sum of per-column counts)
print("Rows with any missing feature:", X_train.isna().any(axis=1).sum())
print("Rows with all features missing:", X_train.isna().all(axis=1).sum())

# Adapt column names to your dataset, e.g. UCI Adult:
# print((X_train["workclass"].isna() & X_train["occupation"].isna()).sum())
```

### Numeric values: summary, range, outliers, suspicious values

```python
display(X_train.describe(include="number").T)
for col in X_train.select_dtypes(include="number"):
    print(col, "zeros:", (X_train[col] == 0).sum(),
          "negative:", (X_train[col] < 0).sum())

# Optional: IQR-based outlier FLAGGING, not automatic deletion
num = X_train.select_dtypes(include="number")
q1, q3 = num.quantile(0.25), num.quantile(0.75)
iqr = q3 - q1
outlier_counts = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum()
display(outlier_counts.sort_values(ascending=False))
```

Zero, negative, or extreme values are **not automatically errors**. Check plausibility and domain meaning. Also check impossible ages, timestamps, or units if relevant.

### Categorical distributions, duplicates, and target balance

```python
for col in X_train.select_dtypes(include=["object", "category", "string"]).columns:
    print(f"\n{col} (top 10)")
    display(X_train[col].value_counts(dropna=False).head(10))

print("Duplicate X rows:", X_train.duplicated().sum())
print("Target counts:")
display(y_train.value_counts(dropna=False))
print("Target fractions:")
display(y_train.value_counts(normalize=True, dropna=False))
```

A duplicate *feature* row need not be a duplicate record; do not drop duplicates automatically. Check whether source records overlap between training/test (especially repeated people, users, or events). For severe class imbalance, prefer per-class precision/recall/F1 over accuracy alone.

## 3. Switching datasets

Only the **loader and target-cleaning block** should normally change. After that, the dtype-based selection automatically identifies numeric and categorical columns.

```python
# Example: a CSV with a named target column
raw = pd.read_csv("your_data.csv", na_values=["?", "NA", "N/A", "null"])
raw.columns = raw.columns.str.strip()
y = raw.pop("target")  # Series; target is excluded from features
X = raw.copy()

# Inspect y.unique(), then encode if doing binary classification:
# y = y.map({"negative_label": 0, "positive_label": 1})
# assert y.notna().all()
```

**Caveats:** Inspect target values (case, whitespace, punctuation, missing labels). Ensure numeric-looking strings are converted to numeric dtypes, and numeric category codes are treated as categories if appropriate. Dtype inference is convenient, **not a guarantee of correct feature semantics**.

## 4. Preprocessing menu: choose only what applies

| Data situation | Choice | Example |
|---|---|---|
| Numeric missing, skew/outliers | Median (good default) | `SimpleImputer(strategy="median")` |
| Numeric missing, fairly symmetric | Mean | `SimpleImputer(strategy="mean")` |
| Numeric/categorical missing, meaningful sentinel | Constant | `SimpleImputer(strategy="constant", fill_value=0)` |
| Categorical missing, most plausible category | Mode | `SimpleImputer(strategy="most_frequent")` |
| Categorical missing, absence may be informative | Separate category | `SimpleImputer(strategy="constant", fill_value="Unknown")` |
| No missing numeric values | No imputer | `StandardScaler()` |
| No missing categorical values | No imputer | `OneHotEncoder(handle_unknown="ignore")` |
| No scaling needed (e.g. many tree models) | Skip scaler | Keep imputer only if needed |
| Numeric distribution has severe outliers | Consider robust scaling | `RobustScaler()` (import separately) |
| A feature is all missing | Investigate / exclude, or use `keep_empty_features=True` | Avoid blindly assigning values |

`SimpleImputer` also supports `strategy="constant"` with a custom `fill_value`, or a **callable** strategy in newer scikit-learn versions. For most beginner notebooks, `mean`, `median`, `most_frequent`, and `constant` cover typical needs. `SimpleImputer(add_indicator=True)` can preserve a binary indicator of *which inputs were missing*; this differs from `OneHotEncoder(handle_unknown="ignore")`, which handles categories unseen during fitting.

**Swappable snippets:**

```python
# Numeric: no missing values
numeric_pipeline = StandardScaler()

# Numeric: median only, no scaling
# numeric_pipeline = SimpleImputer(strategy="median")

# Numeric: impute + scale
# numeric_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
# ])

# Categorical: make missingness its own 'Unknown' category
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# Categorical: mode instead
# categorical_pipeline = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("encoder", OneHotEncoder(handle_unknown="ignore")),
# ])

# Categorical: no missing values
# categorical_pipeline = OneHotEncoder(handle_unknown="ignore")
```

**Feature grouping:**

```python
# Default in main example: auto-detect numeric and categorical dtypes.
# For encoded categories, manually move a column from numeric_cols to categorical_cols.
# Also inspect unsupported dtypes (datetime, timedelta, etc.) before modelling.

# If all numeric or all categorical, omit the unused ColumnTransformer entry:
# preprocessor = ColumnTransformer([("numeric", numeric_pipeline, numeric_cols)])
# preprocessor = ColumnTransformer([("categorical", categorical_pipeline, categorical_cols)])
```

Other missing-data approaches (dropping rows/columns, KNN or iterative imputation) exist, but should be intentional choices. **If you drop rows**, make sure `X` and `y` stay aligned; do not independently drop nulls from each. Prefer a pipeline for learned replacements.

## 5. Model evaluation and threshold choices

For binary problems with `0 = negative`, `1 = positive`:

| Actual \ Predicted | 0 | 1 |
|---|---|---|
| **0** | TN | FP |
| **1** | FN | TP |

- **Accuracy:** `(TP + TN) / total` — can look high on imbalanced data.
- **Precision:** `TP / (TP + FP)` — of predicted positives, how many were right?
- **Recall:** `TP / (TP + FN)` — of actual positives, how many were found?
- **F1:** `2 × precision × recall / (precision + recall)`.
- **Support:** number of actual observations in that class.
- **Confusion matrix:** rows = actual; columns = predicted.

**Choosing scoring:** `scoring="f1"` for binary class `1`; `"f1_macro"` weighs classes equally; `"f1_weighted"` weights by support; `"accuracy"` measures overall correctness. For regression, switch to a regressor and regression metrics instead.

### Change prediction threshold (binary logistic regression)

```python
# Requires a fitted model; does NOT retrain it.
proba_positive = pipeline.predict_proba(X_test)[:, 1]
y_pred_lower = (proba_positive >= 0.30).astype(int)  # default is 0.50
print(classification_report(y_test, y_pred_lower, labels=[0, 1]))
```

Lower threshold → typically more TP **and** more FP → recall rises, precision often falls; F1 may rise or fall. **Do not choose 0.30 from the test set.** Use a separate validation split (or out-of-fold training predictions) to select the threshold; then lock it and evaluate the test set **once**. The snippet above illustrates the mechanism, not an approved procedure for tuning on test data.

## 6. What happens when?

| Operation | Learns anything? |
|---|---|
| Define `Pipeline(...)` | No — only constructs steps |
| `cross_val_score(pipeline, X_train, ...)` | Yes — *separate cloned models* in each fold |
| `pipeline.fit(X_train, y_train)` | Yes — learns preprocessing and model from training data |
| `pipeline.predict(X_test)` | No fitting — applies learned transforms, predicts |
| Threshold comparison on stored probabilities | No retraining — only changes predicted labels |

**Leakage warning:** Never run `imputer.fit_transform(X_train)` or `scaler.fit_transform(X_train)` *before* cross-validation; fold-validation rows would influence fitted parameters. Keep those steps **inside** the pipeline given to CV.

**Split cautions:** `train_test_split(..., stratify=y)` suits typical independent classification examples; use time-aware splits for time series and group-aware splits when samples from the same person/customer/event must remain together. Fixing random seeds is for reproducibility, not a safeguard against leakage.

## 7. Debugging quick reference

| Symptom | Likely fix |
|---|---|
| `could not convert string to float: 'Never-married'` | Ensure categorical columns go through `OneHotEncoder` in `ColumnTransformer` |
| `pos_label=1 is not a valid label` | Normalise and encode target to `0/1`, or use an explicit scorer |
| Target Series has no `.columns` | Use `y.name`, `y.value_counts()`; `.squeeze()` returns a Series |
| `KeyError: 'income'` | Check whether income is in `y` rather than `X`/`df`; inspect column names |
| Unexpected missing values after loading | Audit both true `NaN` and literal placeholders, including spaces/case |
| Train F1 good but validation/test F1 weak | Check overfitting, leakage, unstable splits, and distribution differences |
| One-hot categories in test not in train | `OneHotEncoder(handle_unknown="ignore")` |
| CV F1 ≈ final test F1 | Encouraging consistency, but **not** proof the model is optimal |

## 8. Five-sentence write-up template

1. **Split:** “I reserved 20% as a stratified, untouched test set and trained on the remaining 80%.”
2. **Data quality:** “I inspected types, duplicates, placeholder nulls and class counts; missing values were handled as [method] because [reason].”
3. **Features/model:** “Numeric fields were [imputed/scaled], categorical fields were [imputed/one-hot encoded], and I used [model] as a baseline.”
4. **Leakage:** “I cross-validated the full pipeline, so every fold learned preprocessing from its training subset only.”
5. **Results/errors:** “CV [metric] was [value], final test [metric] was [value], and the main error pattern was [FP vs FN]; I would improve [next step] based on the application's costs.”
