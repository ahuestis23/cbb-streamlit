import pandas as pd
import requests
from io import StringIO

# URLs for datasets
json_url = "https://barttorvik.com/2025_all_advgames.json.gz"
csv_url = "https://barttorvik.com/2025_super_sked.csv"

# Fetch and process JSON data
try:
    response = requests.get(json_url)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
    data = response.json()

    # Define column headers for JSON data
    json_headers = [
        "numdate", "datetext", "opstyle", "quality", "win1", "opponent", "muid", "win2",
        "Min_per", "ORtg", "Usage", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per",
        "TO_per", "dunksmade", "dunksatt", "rimmade", "rimatt", "midmade", "midatt",
        "twoPM", "twoPA", "TPM", "TPA", "FTM", "FTA", "bpm_rd", "Obpm", "Dbpm",
        "bpm_net", "pts", "ORB", "DRB", "AST", "TOV", "STL", "BLK", "stl_per",
        "blk_per", "PF", "possessions", "bpm", "sbpm", "loc", "team", "player", 
        "inches", "cls", "pid", "year"
    ]

    # Convert JSON data to DataFrame
    df_json = pd.DataFrame(data, columns=json_headers)
    df_json['numdate'] = pd.to_datetime(df_json['numdate'], format='%Y%m%d')
    df_json.to_csv("2025_player_game_stats.csv", index=False)
    print("JSON data successfully saved to 2025_player_game_stats.csv")

except requests.exceptions.RequestException as e:
    print(f"Error fetching JSON data: {e}")

# Fetch and process CSV data
try:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(csv_url, headers=headers)
    response.raise_for_status()

    # Define column headers for CSV data
    csv_headers = [
        "muid", "date", "conmatch", "matchup", "prediction", "ttq", "conf", "venue",
        "team1", "t1oe", "t1de", "t1py", "t1wp", "t1propt", "team2", "t2oe", "t2de",
        "t2py", "t2wp", "t2propt", "tpro", "t1qual", "t2qual", "gp", "result",
        "tempo", "possessions", "t1pts", "t2pts", "winner", "loser", "t1adjt",
        "t2adjt", "t1adjo", "t1adjd", "t2adjo", "t2adjd", "gamevalue", "mismatch",
        "blowout", "t1elite", "t2elite", "ord_date", "t1ppp", "t2ppp", "gameppp",
        "t1rk", "t2rk", "t1gs", "t2gs", "gamestats", "overtimes", "t1fun", "t2fun",
        "results"
    ]

    # Convert CSV content to DataFrame with specified headers
    df_csv = pd.read_csv(StringIO(response.text), names=csv_headers, skiprows=1)
    df_csv['date'] = pd.to_datetime(df_csv['date'], format='%m/%d/%y')
    df_csv.to_csv("2025_game_sked.csv", index=False)
    print("CSV data successfully saved to 2025_game_sked.csv")

except requests.exceptions.RequestException as e:
    print(f"Error fetching CSV data: {e}")

