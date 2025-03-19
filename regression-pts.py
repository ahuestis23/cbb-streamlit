import pandas as pd
import joblib

modeling_df = pd.read_csv('2025_merged_data_with_rolling.csv')


# Load datasets
features = [
    'Min_per_l10', 'Usage_l10', 'eFG_l10', 'TS_per_l10', 'ORB_per_l10', 'DRB_per_l10', 'AST_per_l10',
    'TO_per_l10', 'venue', 'team_proj', 'team_adjt', 'team_oe', 'team_de', 
    'opp_proj', 'opp_adjt', 'opp_oe', 'opp_de'
]
target = 'pts'  # We are predicting points

# Drop rows where any of the rolling average columns contain NaN
modeling_df = modeling_df.dropna(subset=[col for col in features if col.endswith('_l10')])

# Select features and target variable
X = modeling_df[features]
y = modeling_df[target]

# Convert boolean columns to numeric (if needed)
X = X.astype(float)

from sklearn.model_selection import train_test_split

# Split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, r2_score

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained model
joblib.dump(rf_model, 'rf_model_pts.pkl')

print("Model saved successfully!")
