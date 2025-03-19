import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import differential_evolution

# Load the CSV file and parse the date column
df = pd.read_csv('2025_merged_data.csv')
df['date'] = pd.to_datetime(df['date'])
df_sorted = df.sort_values('date')

# Sample a fraction of the players (e.g., 20% of unique player IDs)
unique_pids = df_sorted['pid'].unique()
sample_fraction = 0.05  # adjust as needed
sample_pids = np.random.choice(unique_pids, size=int(len(unique_pids) * sample_fraction), replace=False)
df_sample = df_sorted[df_sorted['pid'].isin(sample_pids)]

def objective(beta_value):
    """
    Objective function to compute the mean absolute error (MAE) for a given beta using a sample of players.
    """
    beta = beta_value[0]
    errors = []

    # Loop over each player in the sampled data
    for pid, group in df_sample.groupby('pid'):
        group = group.sort_values('date')
        pts = group['pts'].values
        dates = group['date'].values

        # Skip players with less than 2 games
        if len(group) < 2:
            continue

        # For each game from the second game onward:
        for j in range(1, len(group)):
            current_date = dates[j]
            previous_dates = dates[:j]
            differences = np.array([(current_date - pd.Timestamp(d)).days for d in previous_dates])
            weights = beta ** differences
            weighted_proj = np.sum(pts[:j] * weights) / np.sum(weights)
            errors.append(abs(weighted_proj - pts[j]))

    return np.mean(errors) if errors else np.inf

# Set the bounds for beta (e.g., between 0.90 and 1.0)
bounds = [(0.90, 1.0)]

# Optimize the beta factor using differential evolution on the sampled data
result = differential_evolution(objective, bounds)
optimal_beta = result.x[0]

print("Optimal beta for pts:", optimal_beta)
print("MAE with optimal beta:", result.fun)
