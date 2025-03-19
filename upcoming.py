import pandas as pd
from datetime import datetime
import pytz

# -------------------------------
# Process Game Data
# -------------------------------
# Load game schedule CSV and convert the date column
game_data = pd.read_csv("2025_game_sked.csv")
game_data['date'] = pd.to_datetime(game_data['date'])

# Get today's date in Eastern Time
eastern = pytz.timezone('US/Eastern')
today_eastern = datetime.now(eastern).date()

# Filter for games on or after today
filtered_games = game_data[game_data['date'].dt.date >= today_eastern]

# Include the offensive efficiency columns if they exist, e.g., 't1oe' and 't2oe'
game_cols = ['muid', 'team1', 'team2', 't1de', 't2de', 't1propt', 't2propt', 't1oe', 't2oe']
filtered_games = filtered_games[game_cols]

# -------------------------------
# Process Player Data
# -------------------------------
# Load player projection CSV and filter needed columns
player_data = pd.read_csv("2025_pts_projs.csv")
player_cols = ['pid', 'player', 'team', 'weighted_pts', 'kalman_pts']
player_data = player_data[player_cols].drop_duplicates()

# -------------------------------
# Map Teams to Their Game Information
# -------------------------------
# Create a mapping (dictionary) from each team to its corresponding game row
team_game = {}
for idx, row in filtered_games.iterrows():
    team_game[row['team1']] = row
    team_game[row['team2']] = row

# Function to extract game features for a given team using the actual offensive efficiency values
def get_game_features(team):
    game = team_game.get(team)
    if game is None:
        # If no game is found for this team, return None for all game features
        return pd.Series({
            'team_proj': None,
            'team_oe': None,
            'team_de': None,
            'opp_proj': None,
            'opp_oe': None,
            'opp_de': None
        })
    if team == game['team1']:
        return pd.Series({
            'team_proj': game['t1propt'],
            'team_oe': game['t1oe'],
            'team_de': game['t1de'],
            'opp_proj': game['t2propt'],
            'opp_oe': game['t2oe'],
            'opp_de': game['t2de']
        })
    elif team == game['team2']:
        return pd.Series({
            'team_proj': game['t2propt'],
            'team_oe': game['t2oe'],
            'team_de': game['t2de'],
            'opp_proj': game['t1propt'],
            'opp_oe': game['t1oe'],
            'opp_de': game['t1de']
        })
    else:
        return pd.Series({
            'team_proj': None,
            'team_oe': None,
            'team_de': None,
            'opp_proj': None,
            'opp_oe': None,
            'opp_de': None
        })

# Apply the function to each player's team to get game features
game_features = player_data['team'].apply(get_game_features)
player_data = pd.concat([player_data, game_features], axis=1)

# -------------------------------
# Prepare Final Data Format for the Model
# -------------------------------
# Include 'player' and 'team' along with the model features for reference
final_columns = ['player', 'team', 'weighted_pts', 'kalman_pts', 
                 'team_proj', 'team_oe', 'team_de', 
                 'opp_proj', 'opp_oe', 'opp_de']
final_output = player_data[final_columns]

# Drop any rows that have missing values, which cleans out players with no upcoming game info
final_output = final_output.dropna()
final_output = final_output[final_output['weighted_pts'] != 0]
final_output = final_output[final_output['kalman_pts'] != 0]
#Drop duplicate rows


# Optionally, save to a CSV for later use
final_output.to_csv("model_input.csv", index=False)
print(final_output.head())
