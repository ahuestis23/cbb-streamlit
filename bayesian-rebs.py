import pandas as pd
import numpy as np
from datetime import datetime
from pykalman import KalmanFilter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------
# Data Preparation
# ---------------------------
# Load the CSV file and create the total rebounds column
df = pd.read_csv('2025_merged_data.csv')
df['TRB'] = df['ORB'] + df['DRB']

# Parse the date column
df['date'] = pd.to_datetime(df['date'])
reference_date = datetime(2025, 3, 19)  # update as needed
df['days_since'] = (reference_date - df['date']).dt.days

def compute_decay_weight(days, beta):
    """Compute the exponential decay weight for a given number of days."""
    return beta ** days

# Choose a decay factor for rebounds (you might optimize this later)
beta_reb = 0.996
df['weight'] = compute_decay_weight(df['days_since'], beta_reb)

# ---------------------------
# Weighted Projections for Rebounds
# ---------------------------
# Compute weighted offensive rebounds
weighted_orebs = (
    df.groupby('pid')
      .apply(lambda x: np.sum(x['ORB'] * x['weight']) / np.sum(x['weight']))
      .reset_index(name='weighted_oreb')
)

# Compute weighted defensive rebounds
weighted_drebs = (
    df.groupby('pid')
      .apply(lambda x: np.sum(x['DRB'] * x['weight']) / np.sum(x['weight']))
      .reset_index(name='weighted_dreb')
)

# Merge weighted offensive and defensive projections and sum them to get total weighted rebounds
reb_weighted = pd.merge(weighted_orebs, weighted_drebs, on='pid', how='inner')
reb_weighted['weighted_trb'] = reb_weighted['weighted_oreb'] + reb_weighted['weighted_dreb']

# ---------------------------
# Kalman Filter Projections for Rebounds
# ---------------------------
def apply_kalman_filter(series):
    """
    Apply a Kalman filter to a series.
    Returns the filtered state means.
    """
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=series.iloc[0],
        initial_state_covariance=1,
        observation_covariance=1,
        transition_covariance=0.1
    )
    state_means, _ = kf.filter(series.values)
    return state_means.flatten()

# Ensure data is sorted by date for each player
df_sorted = df.sort_values('date')

# Apply Kalman filter to offensive rebounds and defensive rebounds
kalman_results_oreb = df_sorted.groupby('pid')['ORB'].apply(apply_kalman_filter)
kalman_results_dreb = df_sorted.groupby('pid')['DRB'].apply(apply_kalman_filter)

kalman_projections_orebs = kalman_results_oreb.apply(lambda arr: arr[-1]).reset_index(name='kalman_oreb')
kalman_projections_drebs = kalman_results_dreb.apply(lambda arr: arr[-1]).reset_index(name='kalman_dreb')

# Merge Kalman projections and sum to get total Kalman-based rebounds
reb_kalman = pd.merge(kalman_projections_orebs, kalman_projections_drebs, on='pid', how='inner')
reb_kalman['kalman_trb'] = reb_kalman['kalman_oreb'] + reb_kalman['kalman_dreb']

# ---------------------------
# Merge Projections Back into the Game-Level Data
# ---------------------------
# Merge weighted and Kalman projections on player ID
projections = pd.merge(
    reb_weighted[['pid', 'weighted_trb']],
    reb_kalman[['pid', 'kalman_trb']],
    on='pid', how='inner'
)

# Merge these projections into the main data
df_with_proj = pd.merge(df_sorted, projections, on='pid', how='left')
print(df_with_proj.head())

# Save the projections (optional)
df_with_proj.to_csv('2025_reb_projs.csv', index=False)

# ---------------------------
# Build the Regression Model for Total Rebounds (TRB)
# ---------------------------
# Define the feature list and target.
# (Assuming your game-level data has these additional team context columns)
features = ['weighted_trb', 'kalman_trb', 
            'team_proj', 'team_oe', 'team_de', 
            'opp_proj', 'opp_oe', 'opp_de']
target = 'TRB'

# Drop rows with missing values in the features or target
df_model = df_with_proj.dropna(subset=features + [target])
print(df_model.head())

# Separate features and target variable
X = df_model[features]
y = df_model[target]

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbr_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained model for rebounds
joblib.dump(gbr_model, 'gbr_model_reb.pkl')
