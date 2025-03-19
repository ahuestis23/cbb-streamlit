import pandas as pd

modeling_df = pd.read_csv('2025_merged_data.csv')

# Calculate the average of Ortg, Min_per, and Usage by player_name and team
average_stats = modeling_df.groupby(['player', 'team'])[['ORtg', 'Min_per', 'Usage']].mean().reset_index()

average_stats.to_csv('average_stats.csv', index=False)
