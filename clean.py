import pandas as pd

# Load datasets
player_data = pd.read_csv("2025_player_game_stats.csv")
game_data = pd.read_csv("2025_game_sked.csv")

# Only keep the columns you need for the analysis
player_cols = [
    "muid", "pid", "player", "team", "opponent", "Min_per", "ORtg", "Usage", 
    "eFG", 
    "TS_per", "ORB_per", "DRB_per", "AST_per",
    "TO_per", "dunksmade", "dunksatt", "rimmade", "rimatt", "midmade", "midatt",
    "twoPM", "twoPA", "TPM", "TPA", "FTM", "FTA", "bpm_rd", "Obpm", "Dbpm",
    "bpm_net", "pts", "ORB", "DRB", "AST", "TOV", "STL", "BLK", "stl_per",
    "blk_per", "PF", "bpm", "sbpm", "inches",
    "cls", "year"
]

#removed 'prediction', maybe bring back in later?
game_cols = [
    "muid", "date", "ttq", "conf", "venue",
    "team1", "t1oe", "t1de", "t1py", "t1propt", "team2", "t2oe", "t2de",
    "t2py", "t2propt", "tpro", "tempo", "possessions", "t1pts", "t2pts", "t1adjt","t2adjt", 
    "t1adjo", "t1adjd", "t2adjo", "t2adjd", "gamevalue", 
    "mismatch", "blowout", "t1elite", "t2elite", "t1ppp", "t2ppp", "gameppp",
    "t1rk", "t2rk", "t1gs", "t2gs", "overtimes"
]

# Filter player data to include only the columns you need
player_data = player_data[player_cols]
# Filter game data to include only the columns you
game_data = game_data[game_cols]

# Fill NaN values in numeric columns with 0
player_data.fillna(0, inplace=True)
game_data.fillna(0, inplace=True)

# Merge on Date and Team (Ensure both match for either team1 or team2)
merged_df = player_data.merge(game_data, 
                              left_on=['muid'], 
                              right_on=['muid'], 
                              how='left')

# Convert to percentages instead of floats
percentage_cols = ['TS_per', 'Usage', 'eFG', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per', 'stl_per', 'blk_per']
merged_df[percentage_cols] = merged_df[percentage_cols] / 100

# Create new columns for team
merged_df["team_oe"] = merged_df.apply(lambda x: x["t1oe"] if x["team"] == x["team1"] else x["t2oe"], axis=1)
merged_df["team_de"] = merged_df.apply(lambda x: x["t1de"] if x["team"] == x["team1"] else x["t2de"], axis=1)
merged_df["team_py"] = merged_df.apply(lambda x: x["t1py"] if x["team"] == x["team1"] else x["t2py"], axis=1)
merged_df["team_proj"] = merged_df.apply(lambda x: x["t1propt"] if x["team"] == x["team1"] else x["t2propt"], axis=1)
merged_df["team_pts"] = merged_df.apply(lambda x: x["t1pts"] if x["team"] == x["team1"] else x["t2pts"], axis=1)
merged_df["team_adjt"] = merged_df.apply(lambda x: x["t1adjt"] if x["team"] == x["team1"] else x["t2adjt"], axis=1)
merged_df["team_adjo"] = merged_df.apply(lambda x: x["t1adjo"] if x["team"] == x["team1"] else x["t2adjo"], axis=1)
merged_df["team_adjd"] = merged_df.apply(lambda x: x["t1adjd"] if x["team"] == x["team1"] else x["t2adjd"], axis=1)
merged_df["team_elite"] = merged_df.apply(lambda x: x["t1elite"] if x["team"] == x["team1"] else x["t2elite"], axis=1)
merged_df["team_ppp"] = merged_df.apply(lambda x: x["t1ppp"] if x["team"] == x["team1"] else x["t2ppp"], axis=1)
merged_df["team_rank"] = merged_df.apply(lambda x: x["t1rk"] if x["team"] == x["team1"] else x["t2rk"], axis=1)
merged_df["team_gs"] = merged_df.apply(lambda x: x["t1gs"] if x["team"] == x["team1"] else x["t2gs"], axis=1)

# Create new columns for opponent stats based on the matchup
merged_df["opp_oe"] = merged_df.apply(lambda x: x["t2oe"] if x["team"] == x["team1"] else x["t1oe"], axis=1)
merged_df["opp_de"] = merged_df.apply(lambda x: x["t2de"] if x["team"] == x["team1"] else x["t1de"], axis=1)
merged_df["opp_py"] = merged_df.apply(lambda x: x["t2py"] if x["team"] == x["team1"] else x["t1py"], axis=1)
merged_df["opp_proj"] = merged_df.apply(lambda x: x["t2propt"] if x["team"] == x["team1"] else x["t1propt"], axis=1)
merged_df["opp_pts"] = merged_df.apply(lambda x: x["t2pts"] if x["team"] == x["team1"] else x["t1pts"], axis=1)
merged_df["opp_adjt"] = merged_df.apply(lambda x: x["t2adjt"] if x["team"] == x["team1"] else x["t1adjt"], axis=1)
merged_df["opp_adjo"] = merged_df.apply(lambda x: x["t2adjo"] if x["team"] == x["team1"] else x["t1adjo"], axis=1)
merged_df["opp_adjd"] = merged_df.apply(lambda x: x["t2adjd"] if x["team"] == x["team1"] else x["t1adjd"], axis=1)
merged_df["opp_elite"] = merged_df.apply(lambda x: x["t2elite"] if x["team"] == x["team1"] else x["t1elite"], axis=1)
merged_df["opp_ppp"] = merged_df.apply(lambda x: x["t2ppp"] if x["team"] == x["team1"] else x["t1ppp"], axis=1)
merged_df["opp_rank"] = merged_df.apply(lambda x: x["t2rk"] if x["team"] == x["team1"] else x["t1rk"], axis=1)
merged_df["opp_gs"] = merged_df.apply(lambda x: x["t2gs"] if x["team"] == x["team1"] else x["t1gs"], axis=1)

columns_to_drop = [
    "t1oe", "t1de", "t1py", "t1propt", "t1pts", "t1adjt", "t1adjo", "t1adjd", "t1elite", "t1ppp", "t1rk", "t1gs",
    "t2oe", "t2de", "t2py", "t2propt", "t2pts", "t2adjt", "t2adjo", "t2adjd", "t2elite", "t2ppp", "t2rk", "t2gs",
    "team1", "team2"
]

merged_df = merged_df.drop(columns=columns_to_drop, errors="ignore")

# Convert 'date' column to datetime format for proper sorting
merged_df["date"] = pd.to_datetime(merged_df["date"])

# Sort by player ID (pid), match ID (muid), and date to ensure correct rolling calculations
merged_df = merged_df.sort_values(by=["pid", "muid", "date"])

rolling_df = merged_df.copy()
# Define the columns for which to calculate the rolling average
rolling_cols = ['Min_per', 'Usage', 'eFG', 'TS_per', 'ORB_per', 'DRB_per', 'AST_per', 'TO_per']

# Calculate the rolling mean for the last 10 games (excluding the current game)
for col in rolling_cols:
    rolling_df[f"{col}_l10"] = rolling_df.groupby(["pid"])[col].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())

# Save the updated dataset
rolling_df.to_csv("2025_merged_data_with_rolling.csv", index=False)

merged_df.to_csv("2025_merged_data.csv", index=False)
#merged_df.head().to_csv("2025_merged_data_head.csv", index=False)