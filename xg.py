import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# 📌 **Step 1: Load & Clean Data**
df = pd.read_csv("2025_merged_data_with_rolling.csv")

# 🔹 **Step 2: Feature Selection**
features = [
    'Min_per_l10', 'Usage_l10', 'eFG_l10', 'TS_per_l10', 'ORB_per_l10', 'DRB_per_l10', 'AST_per_l10',
    'TO_per_l10', 'venue', 'team_proj', 'team_adjt', 'team_oe', 'team_de', 
    'opp_proj', 'opp_adjt', 'opp_oe', 'opp_de'
]
target = 'pts'  # We are predicting points

# 🔹 **Step 3: Handle Missing Values (Drop first-game NaNs)**
df = df.dropna(subset=[col for col in features if col.endswith('_l10')])

# 🔹 **Step 4: Split into Train/Test Sets**
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 📌 **Step 5: Train Baseline XGBoost Model**
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# 🔹 **Step 6: Make Predictions**
y_pred = xgb_model.predict(X_test)

# 🔹 **Step 7: Evaluate Performance**
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost Mean Absolute Error: {mae:.2f}")
print(f"XGBoost R² Score: {r2:.2f}")

# 🔹 **Step 8: Hyperparameter Tuning (Optional)**
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

xgb = XGBRegressor(random_state=42)
random_search = RandomizedSearchCV(
    xgb, param_distributions=param_grid, n_iter=20, cv=5, scoring='r2', n_jobs=-1, verbose=2
)
random_search.fit(X_train, y_train)

# 🔹 **Step 9: Train Best Model**
best_xgb = random_search.best_estimator_
best_xgb.fit(X_train, y_train)

# 🔹 **Step 10: Final Predictions**
y_pred_best = best_xgb.predict(X_test)

# 🔹 **Step 11: Final Model Evaluation**
final_mae = mean_absolute_error(y_test, y_pred_best)
final_r2 = r2_score(y_test, y_pred_best)

print(f"Optimized XGBoost Mean Absolute Error: {final_mae:.2f}")
print(f"Optimized XGBoost R² Score: {final_r2:.2f}")

# 📌 **Step 12: Save the Model**
joblib.dump(best_xgb, 'xgb_model_pts.pkl')
print("Optimized XGBoost model saved successfully!")
