import pandas as pd
import numpy as np
from datetime import datetime

# Load your CSV file (update the path if needed)
df = pd.read_csv('2025_merged_data.csv')

# Ensure that the game date is parsed as a datetime object.
# (Replace 'game_date' with the actual column name if different.)
df['date'] = pd.to_datetime(df['date'])

# Choose a reference date (today)
reference_date = datetime.today()

# Calculate the number of days since each game
df['days_since'] = (reference_date - df['date']).dt.days

def compute_decay_weight(days, beta):
  """
  Compute the exponential decay weight for a given number of days.
  """
  return beta ** days

# Choose an initial decay factor for points (you may optimize this later)
beta_points = 0.99

# Compute the decay weight for each game based on days_since
df['weight_pts'] = compute_decay_weight(df['days_since'], beta_points)

# To compute a weighted average points projection per player (for example):
weighted_points = (
  df.groupby('pid')
    .apply(lambda x: np.sum(x['pts'] * x['weight_pts']) / np.sum(x['weight_pts']))
    .reset_index(name='weighted_pts')
)

from pykalman import KalmanFilter

def apply_kalman_filter(series):
    """
    Apply a Kalman filter to a series.
    Returns the filtered state means.
    """
    kf = KalmanFilter(
        transition_matrices = [1],
        observation_matrices = [1],
        initial_state_mean = series.iloc[0],
        initial_state_covariance = 1,
        observation_covariance = 1,
        transition_covariance = 0.1
    )
    state_means, _ = kf.filter(series.values)
    return state_means.flatten()

# Ensure the data is sorted by date for each player
df_sorted = df.sort_values('date')

# Apply the Kalman filter for 'pts'
kalman_results_pts = df_sorted.groupby('pid')['pts'].apply(apply_kalman_filter)

kalman_projections_pts = kalman_results_pts.apply(lambda arr: arr[-1]).reset_index(name='kalman_pts')

# Merge the weighted points projections (already computed earlier) with the Kalman projections for points
projections = pd.merge(weighted_points, kalman_projections_pts, on='pid', how='inner')

# Merge the projections back into the original game-level data.
df_with_proj = pd.merge(df_sorted, projections, on='pid', how='left')
print(df_with_proj.head())
df_with_proj.to_csv('2025_pts_projs.csv', index=False)

# Example feature list. Update these to match the actual column names in your data.
features = ['weighted_pts', 'kalman_pts', 'team_proj',  'team_oe', 'team_de','opp_proj', 'opp_oe', 'opp_de'] 
target = 'pts'

# Drop rows with missing values for the selected features
df_model = df_with_proj.dropna(subset=features + [target])
print(df_model.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Separate features and target variable
X = df_model[features]
y = df_model[target]

# Split the data into training and test sets (e.g., 80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbr_model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")


import joblib
joblib.dump(gbr_model, 'gbr_model_pts.pkl')
