# Used Car Price Prediction — Project Report

**Dataset:** Vehicle Dataset from CarDekho ([Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), `Car details v3.csv`)
**Goal:** Predict used car selling price (Indian market) from listing attributes using a full, leakage-safe machine learning workflow.

---

## 1. Motivation

This project applies an end-to-end supervised learning workflow (stratified splitting, EDA discipline, pipeline-based preprocessing, cross-validation, hyperparameter tuning) to a real-world, messy dataset. The dataset offers genuine challenges: a skewed target, inconsistent units across technical spec columns, high-cardinality categoricals, and string fields requiring careful parsing — making it a good testbed for demonstrating disciplined data science practice end to end.

**Note on data currency:** This dataset was collected around 2020 and reflects the Indian used-car market (prices in INR) at that time. It is used here purely for methodology practice, not as a production-ready pricing tool — inflation, currency shifts, and post-2020 market changes are not reflected.

---

## 2. Setup & Data Loading

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/Car details v3.csv")
df.shape
df.head()
df.describe()
df.info()
```

```
<class 'pandas.DataFrame'>
RangeIndex: 8128 entries, 0 to 8127
Data columns (total 13 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   name           8128 non-null   str    
 1   year           8128 non-null   int64  
 2   selling_price  8128 non-null   int64  
 3   km_driven      8128 non-null   int64  
 4   fuel           8128 non-null   str    
 5   seller_type    8128 non-null   str    
 6   transmission   8128 non-null   str    
 7   owner          8128 non-null   str    
 8   mileage        7907 non-null   str    
 9   engine         7907 non-null   str    
 10  max_power      7913 non-null   str    
 11  torque         7906 non-null   str    
 12  seats          7907 non-null   float64
dtypes: float64(1), int64(3), str(9)
memory usage: 825.6 KB
```

**Source choice:** `Car details v3.csv` was used instead of the smaller files in the same Kaggle dataset (`car data.csv`, `CAR DETAILS FROM CAR DEKHO.csv`), for its larger size (8,128 rows) and richer, string-encoded technical specs (`mileage`, `engine`, `max_power`, `torque`) requiring real feature engineering.

---

## 3. Handling Missing Data

Five columns (`mileage`, `engine`, `max_power`, `torque`, `seats`) were missing in the same ~222 rows (~2.7% of data) — a blocked pattern suggesting incomplete listings rather than random gaps. These rows were dropped rather than imputed, since imputing five correlated, car-specific technical specs simultaneously would fabricate too much of a car's profile.

```python
missing_mask = df[['mileage', 'engine', 'max_power', 'torque', 'seats']].isnull()
df[missing_mask.any(axis=1)].shape[0]   # confirms 222 overlapping rows

df = df.dropna(subset=['mileage', 'engine', 'max_power', 'torque', 'seats'])
df.shape
```

```
(7906, 13)
```

---

## 4. Extracting Brand from Name

```python
df['brand'] = df['name'].str.split(' ').str[0]
df['brand'] = df['brand'].replace({'Land': 'Land Rover', 'Ashok': 'Ashok Leyland'})
df['brand'].value_counts()
```

```
brand
Maruti           2367
Hyundai          1360
Mahindra          758
Tata              719
Honda             466
Toyota            452
Ford              388
Chevrolet         230
Renault           228
Volkswagen        185
BMW               118
Skoda             104
Nissan             81
Jaguar             71
Volvo              67
Datsun             65
Mercedes-Benz      54
Fiat               41
Audi               40
Lexus              34
Jeep               31
Mitsubishi         14
Land Rover          6
Force               6
Isuzu               5
Kia                 4
Ambassador          4
MG                  3
Daewoo              3
Ashok Leyland       1
Opel                1
Name: count, dtype: int64
```

The two-word brands `Land Rover` and `Ashok Leyland` were mis-split by the naive first-token extraction (`'Land'`, `'Ashok'`) and corrected manually.

---

## 5. Stratified Train/Test Split

`fuel` was heavily imbalanced (Diesel ~54%, Petrol ~45%, CNG ~0.7%, LPG ~0.4%). A random split risked misrepresenting the rare CNG/LPG categories in either set, so the split was stratified on `fuel`.

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(
    df, test_size=0.2, stratify=df['fuel'], random_state=42
)

train_set['fuel'].value_counts(normalize=True)
```

```
fuel
Diesel    0.543616
Petrol    0.445006
CNG       0.006953
LPG       0.004425
Name: proportion, dtype: float64
```

---

## 6. Target Distribution & Log Transform

`selling_price` was strongly right-skewed (mean ≫ median, max = 10M INR).

```python
plt.figure(figsize=(10, 5))
sns.histplot(train_set['selling_price'], bins=50, kde=True)
plt.title('Distribution of Selling Price')
plt.show()

train_set['selling_price_log'] = np.log1p(train_set['selling_price'])

plt.figure(figsize=(10, 5))
sns.histplot(train_set['selling_price_log'], bins=50, kde=True)
plt.title('Distribution of Log-Transformed Selling Price')
plt.show()
```

![Selling price distribution](images/selling_price_dist.png)
![Log-transformed selling price distribution](images/selling_price_log_dist.png)

`log1p(selling_price)` was used as the modeling target throughout the rest of the project.

---

## 7. km_driven Outlier Investigation

```python
plt.figure(figsize=(10, 5))
sns.histplot(train_set['km_driven'], bins=50, kde=True)
plt.title('Distribution of Km Driven')
plt.show()

train_set['km_driven'].sort_values(ascending=False).head(10)
train_set.iloc[[3486, 1810]]   # inspect the two extreme rows

train_set = train_set[train_set['km_driven'] <= 700000]
```

![km_driven distribution before cleaning](images/km_driven_dist.png)

Two listings (2,360,457 km and 1,500,000 km) were investigated against their car profile (age, brand, fuel type) and judged physically implausible — both would require roughly 500 km/day of sustained use. Rather than removing only these two rows by index, a general value-based filter (`km_driven ≤ 700,000`) was applied for robustness against re-running cells out of order.

---

## 8. Deriving Car Age

```python
train_set['car_age'] = 2020 - train_set['year']
train_set[['year', 'car_age']].describe()
```

```
              year     car_age
count  6323.000000  6323.000000
mean   2013.999051     6.000949
std       3.877024     3.877024
min    1994.000000     0.000000
25%    2012.000000     3.000000
50%    2015.000000     5.000000
75%    2017.000000     8.000000
max    2020.000000    26.000000
```

Reference year 2020 was used (matching the dataset's approximate collection period) rather than the current calendar year, so that `car_age` reflects the listing's actual context.

---

## 9. Correlation Matrix — Core Numeric Features

```python
numeric_cols = ['selling_price_log', 'car_age', 'km_driven', 'seats']
corr_matrix = train_set[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Correlation Matrix')
plt.show()
```

![Core correlation matrix](images/correlation_matrix_core.png)

`car_age` showed a strong negative correlation with log price (-0.71); `km_driven` was moderately correlated with `car_age` (0.49), foreshadowing the multicollinearity discussion in §14.

---

## 10. Price by Brand and Categorical Variables

```python
train_set.groupby('brand')['selling_price'].median().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
order = train_set.groupby('brand')['selling_price_log'].median().sort_values(ascending=False).index
sns.boxplot(data=train_set, x='brand', y='selling_price_log', order=order)
plt.xticks(rotation=90)
plt.title('Selling Price (log) by Brand')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=train_set, x='fuel', y='selling_price_log', ax=axes[0])
sns.boxplot(data=train_set, x='transmission', y='selling_price_log', ax=axes[1])
sns.boxplot(data=train_set, x='owner', y='selling_price_log', ax=axes[2])
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
```

![Selling price by brand](images/price_by_brand_boxplot.png)
![Selling price by fuel, transmission, and owner](images/price_by_categorical.png)

Two notable EDA findings shaped later decisions:
- **Brand showed a strong visual price gradient** (BMW/Lexus/Land Rover at the top, Ambassador/Daewoo/Opel at the bottom) — this motivated the brand segmentation in §15, though it was later found to be largely a confounding effect (§19).
- **`owner = "Test Drive Car"` priced *above* "First Owner"** — this category represents near-zero-km dealer demo vehicles, not a step in the ownership chain. This informed a custom ordinal ordering (`Test Drive Car < First < Second < Third < Fourth & Above`) rather than a naive alphabetical one, used in the pipeline (§16).

---

## 11. Parsing Mileage

`mileage` mixed two incompatible units: `kmpl` (Diesel/Petrol) and `km/kg` (CNG/LPG). Rather than merging non-comparable units, rows using `km/kg` were dropped.

```python
train_set = train_set[~train_set['mileage'].str.contains('km/kg', na=False)]
train_set['mileage_numeric'] = train_set['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
train_set = train_set[train_set['mileage_numeric'] != 0]
train_set['mileage_numeric'].describe()
```

```
count    6242.000000
mean       19.408925
std         3.921208
min         9.000000
25%        16.780000
50%        19.300000
75%        22.320000
max        42.000000
Name: mileage_numeric, dtype: float64
```

A handful of rows also had `mileage_numeric == 0` (a hidden missing-value encoding, not a real reading) and were dropped as well.

**Known limitation:** dropping `km/kg` rows removed nearly all CNG/LPG vehicles from the training data — see §21.

---

## 12. Parsing Engine and Max Power

```python
train_set['engine_numeric'] = train_set['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
train_set['max_power_numeric'] = train_set['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)
```

Both columns used a single consistent unit (CC and bhp respectively) and required no further cleaning.

---

## 13. Parsing Torque (Unit Normalization + Error Correction)

`torque` mixed `Nm` and `kgm` units in inconsistent capitalization, plus embedded RPM values.

```python
train_set['torque_numeric'] = train_set['torque'].str.extract(r'(\d+\.?\d*)').astype(float)
train_set['torque_unit'] = train_set['torque'].str.extract(r'(?i)(nm|kgm)')[0].str.lower()

train_set['torque_nm'] = train_set.apply(
    lambda row: row['torque_numeric'] * 9.80665 if row['torque_unit'] == 'kgm' else row['torque_numeric'],
    axis=1
)

# Correct data-entry artifacts (e.g. 789 -> 78.9)
train_set.loc[train_set['torque_nm'] > 700, 'torque_nm'] = train_set.loc[train_set['torque_nm'] > 700, 'torque_nm'] / 10

train_set['torque_nm'].describe()
```

```
count    6242.000000
mean      178.820772
std        91.115504
min        47.071920
25%       113.000000
50%       171.808187
75%       209.000000
max       640.000000
Name: torque_nm, dtype: float64
```

Mixed units were unified via the physical conversion 1 kgm = 9.80665 Nm. A subsequent sanity check against the resulting maximum caught a data-entry artifact: three "Maruti Zen D" rows showed 789 Nm, physically implausible for that model (real-world spec ≈ 78–79 Nm). This was corrected via a decimal-shift heuristic (`/10`) rather than dropped, preserving those rows' otherwise-valid data.

---

## 14. Correlation Matrix — Engineered Features

```python
numeric_cols_full = [
    'selling_price_log', 'car_age', 'km_driven', 'seats',
    'mileage_numeric', 'engine_numeric', 'max_power_numeric', 'torque_nm'
]
corr_matrix_full = train_set[numeric_cols_full].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_full, annot=True, cmap='coolwarm', fmt='.2f', center=0)
plt.title('Correlation Matrix (Extended)')
plt.show()
```

![Extended correlation matrix](images/correlation_matrix_extended.png)

`engine_numeric`, `max_power_numeric`, and `torque_nm` showed strong pairwise correlation (0.70–0.84) — expected, since these are mechanically related. This multicollinearity was tolerated rather than resolved by dropping features, since the planned model (Random Forest) is robust to it; see §17 for the comparison against Linear Regression, which is more sensitive to this issue.

---

## 15. Consolidating Brands into Price Segments

One-hot-encoding all 31 raw brand values would create excessive sparse columns (several brands had only 1–5 rows). Rather than bucketing rare brands into a generic "Other" category, the 31 brands were consolidated into 4 segments using natural breaks in median brand price — preserving market-positioning signal even for low-frequency brands like Opel and Daewoo.

```python
brand_segment_map = {
    'BMW': 'Premium', 'Lexus': 'Premium', 'Land Rover': 'Premium',
    'Jaguar': 'Premium', 'Volvo': 'Premium', 'Mercedes-Benz': 'Premium',
    'Audi': 'Premium', 'Isuzu': 'Premium', 'MG': 'Premium', 'Jeep': 'Premium',
    'Kia': 'Premium',
    'Force': 'Upper-Mid', 'Mitsubishi': 'Upper-Mid', 'Toyota': 'Upper-Mid',
    'Skoda': 'Upper-Mid', 'Mahindra': 'Upper-Mid', 'Honda': 'Upper-Mid',
    'Ford': 'Upper-Mid',
    'Hyundai': 'Lower-Mid', 'Volkswagen': 'Lower-Mid', 'Maruti': 'Lower-Mid',
    'Renault': 'Lower-Mid', 'Nissan': 'Lower-Mid', 'Ashok Leyland': 'Lower-Mid',
    'Datsun': 'Lower-Mid', 'Tata': 'Lower-Mid', 'Fiat': 'Lower-Mid',
    'Chevrolet': 'Lower-Mid',
    'Ambassador': 'Budget', 'Daewoo': 'Budget', 'Opel': 'Budget'
}

train_set['brand_segment'] = train_set['brand'].map(brand_segment_map)
train_set['brand_segment'].isnull().sum()
train_set['brand_segment'].value_counts()
```

```
brand_segment
Lower-Mid    4115
Upper-Mid    1761
Premium       358
Budget          8
Name: count, dtype: int64
```

`isnull().sum()` returned 0, confirming every brand in the training data was successfully mapped.

**Caveat:** segment boundaries were chosen by visually inspecting gaps in median brand price rather than a formal clustering method — a reasonable but somewhat subjective simplification.

---

## 16. Preprocessing Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

numeric_features = ['car_age', 'km_driven', 'mileage_numeric', 'engine_numeric',
                     'max_power_numeric', 'torque_nm', 'seats']
nominal_features = ['fuel', 'seller_type', 'transmission', 'brand_segment']
ordinal_features = ['owner']
owner_order = [['Test Drive Car', 'First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
    ('ord', OrdinalEncoder(categories=owner_order), ordinal_features)
])

feature_cols = numeric_features + nominal_features + ordinal_features
X_train = train_set[feature_cols]
y_train = train_set['selling_price_log']

X_train_transformed = preprocessor.fit_transform(X_train)
X_train_transformed.shape
```

```
(6242, 19)
```

---

## 17. Model Comparison: Linear Regression vs Random Forest

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

lin_reg_pipeline = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])
rf_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))])

lin_scores = cross_val_score(lin_reg_pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
rf_scores = cross_val_score(rf_pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)

print("Linear Regression RMSE:", -lin_scores.mean(), "+/-", lin_scores.std())
print("Random Forest RMSE:", -rf_scores.mean(), "+/-", rf_scores.std())
```

```
Linear Regression RMSE: 0.28863678993175856 +/- 0.002855986459026049
Random Forest RMSE: 0.20926474624625685 +/- 0.007124551403490636
```

Random Forest outperformed Linear Regression by ~28% (RMSE, log scale), consistent with non-linear age-depreciation effects and the multicollinearity noted in §14, which Linear Regression handles poorly but tree ensembles tolerate well.

---

## 18. Hyperparameter Tuning (GridSearchCV)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)
```

```
Best params: {'model__max_depth': 20, 'model__min_samples_split': 5, 'model__n_estimators': 300}
Best RMSE: 0.20597557862835708
```

Tuning improved CV RMSE only marginally (0.2093 → 0.2060), indicating the default Random Forest was already close to optimal for this feature set.

---

## 19. Feature Importance

```python
best_model = grid_search.best_estimator_
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
importances = best_model.named_steps['model'].feature_importances_

importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}) \
    .sort_values('importance', ascending=False)
importance_df
```

| Feature | Importance |
|---|---|
| `num__max_power_numeric` | 0.4098 |
| `num__car_age` | 0.3947 |
| `num__torque_nm` | 0.1166 |
| `num__mileage_numeric` | 0.0217 |
| `num__km_driven` | 0.0205 |
| `num__engine_numeric` | 0.0160 |
| `num__seats` | 0.0057 |
| `ord__owner` | 0.0051 |
| `nom__brand_segment_Premium` | 0.0028 |
| `nom__brand_segment_Lower-Mid` | 0.0024 |
| `nom__brand_segment_Upper-Mid` | 0.0010 |
| `nom__seller_type_Individual` | 0.0008 |
| `nom__seller_type_Dealer` | 0.0008 |
| `nom__transmission_Manual` | 0.0006 |
| `nom__transmission_Automatic` | 0.0006 |
| `nom__fuel_Petrol` | 0.0004 |
| `nom__fuel_Diesel` | 0.0003 |
| `nom__seller_type_Trustmark Dealer` | 0.00003 |
| `nom__brand_segment_Budget` | 0.00001 |

**Key finding:** three numeric engineering features (`max_power`, `car_age`, `torque_nm`) account for ~92% of predictive power. `brand_segment` — despite showing a visually striking price gradient in the EDA boxplot (§10) — contributed under 0.7% combined. This suggests brand's apparent EDA relationship with price was largely a **confounding effect**: premium brands correlate with higher power and newer age, but brand itself carries little independent signal once those are accounted for.

---

## 20. Applying Feature Engineering to the Test Set & Evaluating

All feature engineering steps from §4–§15 were reapplied to the held-out test set — transform only, no re-fitting of any statistic (e.g. `brand_segment_map` was reused as learned from training data).

```python
test_set['brand'] = test_set['name'].str.split(' ').str[0]
test_set['brand'] = test_set['brand'].replace({'Land': 'Land Rover', 'Ashok': 'Ashok Leyland'})
test_set['car_age'] = 2020 - test_set['year']

test_set = test_set[~test_set['mileage'].str.contains('km/kg', na=False)]
test_set['mileage_numeric'] = test_set['mileage'].str.extract(r'(\d+\.?\d*)').astype(float)
test_set = test_set[test_set['mileage_numeric'] != 0]

test_set['engine_numeric'] = test_set['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
test_set['max_power_numeric'] = test_set['max_power'].str.extract(r'(\d+\.?\d*)').astype(float)

test_set['torque_numeric'] = test_set['torque'].str.extract(r'(\d+\.?\d*)').astype(float)
test_set['torque_unit'] = test_set['torque'].str.extract(r'(?i)(nm|kgm)')[0].str.lower()
test_set['torque_nm'] = test_set.apply(
    lambda row: row['torque_numeric'] * 9.80665 if row['torque_unit'] == 'kgm' else row['torque_numeric'],
    axis=1
)
test_set.loc[test_set['torque_nm'] > 700, 'torque_nm'] = test_set.loc[test_set['torque_nm'] > 700, 'torque_nm'] / 10

test_set = test_set[test_set['km_driven'] <= 700000]
test_set['brand_segment'] = test_set['brand'].map(brand_segment_map)
test_set['selling_price_log'] = np.log1p(test_set['selling_price'])
test_set = test_set.dropna(subset=['mileage', 'engine', 'max_power', 'torque', 'seats'])

X_test = test_set[feature_cols]
y_test = test_set['selling_price_log']

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = best_model.predict(X_test)
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
```

```
Test RMSE: 0.1939
Test MAE: 0.1337
Test R²: 0.9402
```

| Metric | Cross-Validation | Test Set |
|---|---|---|
| RMSE (log scale) | 0.2060 | 0.1939 |
| MAE (log scale) | — | 0.1337 |
| R² | — | 0.9402 |

Test performance matched (slightly exceeded) cross-validation performance, indicating **no overfitting**. R² = 0.94 means the model explains 94% of variance in log-price on unseen data.

---

## 21. Prediction Error Distribution

```python
comparison = pd.DataFrame({
    'actual_price': np.expm1(y_test.values),
    'predicted_price': np.expm1(y_pred)
})
comparison['error_pct'] = (comparison['predicted_price'] - comparison['actual_price']) / comparison['actual_price'] * 100
comparison['error_pct'].describe()
```

```
count    1558.000000
mean        1.643625
std        21.440280
min       -60.708850
25%        -9.270658
50%        -0.000000
75%         9.345253
max       211.773300
Name: error_pct, dtype: float64
```

- **Mean ≈ +1.6%**: no strong systematic bias (model does not consistently over- or under-price).
- **Median ≈ 0%**: the typical prediction is very close to the actual price.
- **IQR: −9.3% to +9.3%**: half of all predictions fall within roughly ±10% of the true price.
- **Range: −60.7% to +211.8%**: a minority of predictions — likely rare or extreme feature combinations underrepresented in training — are significantly off.

---

## 22. Real-World Prediction Check

A real 2018 Maruti Swift VDi (Diesel) was priced using its actual published specifications (74 bhp, 190 Nm, 1248cc, 28.4 kmpl mileage).

```python
sample_car = pd.DataFrame({
    'car_age': [2], 'km_driven': [35000], 'mileage_numeric': [28.4],
    'engine_numeric': [1248], 'max_power_numeric': [74], 'torque_nm': [190],
    'seats': [5.0], 'fuel': ['Diesel'], 'seller_type': ['Individual'],
    'transmission': ['Manual'], 'brand_segment': ['Lower-Mid'], 'owner': ['First Owner']
})

predicted_log_price = best_model.predict(sample_car)
predicted_price = np.expm1(predicted_log_price)
print(f"Predicted price (INR): {predicted_price[0]:,.0f}")
```

```
Predicted price (INR): 729,947
```

This is roughly 12–15% above typical real-world market listings for this configuration (₹450,000–650,000) — a gap consistent with, and explained by, the error spread quantified in §21 (half of all test predictions fall within ±10%, but individual predictions can deviate further).

---

## 23. Known Limitations

1. **CNG/LPG vehicles excluded from training.** Dropping `km/kg`-unit `mileage` rows (§11) removed nearly all CNG/LPG listings from the training set, despite an initial stratified split designed to preserve them. The model should be considered valid for **Diesel and Petrol vehicles only**.
2. **Temporal/currency scope.** Data reflects ~2020 Indian market pricing in INR; not adjusted for inflation or currency, and not intended for present-day valuation.
3. **`brand_segment` boundaries were manually chosen** by visually inspecting gaps in median brand price rather than a formal clustering method — a reasonable but somewhat subjective simplification.
4. **Wide error tails.** While typical errors are small (~±10%), a minority of predictions deviate by 50%+ from actual price, suggesting the model is less reliable for atypical feature combinations underrepresented in training data.

---

## 24. Conclusion

This project applied a disciplined, leakage-safe ML workflow (stratified splitting, EDA restricted to the training set, pipeline-based preprocessing, cross-validated model comparison, hyperparameter tuning) to a messy, real-world dataset requiring substantial feature engineering (unit normalization, string parsing, data-entry error detection, cardinality reduction). The tuned Random Forest achieved R² = 0.94 on held-out test data with no evidence of overfitting. The most valuable methodological takeaway was empirical: an EDA-driven hypothesis (brand as a key price driver) was tested and largely refuted once feature importance was measured on the tuned model — a concrete illustration of correlation vs. confounding, and of why EDA conclusions should be revisited once real model diagnostics are available.

---

*Prepared as a portfolio project — Arda, Université de Strasbourg*
