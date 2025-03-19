import streamlit as st
import pandas as pd
import joblib
from datetime import date

st.title("Team Projections")

# Create four tabs: Points, Assists, Rebounds, and Notebook
tabs = st.tabs(["Points Projections", "Assists Projections", "Rebounds Projections", "Notebook"])

# Global team selection (applies to the first three tabs)
@st.cache_data
def load_input_data_pts():
    return pd.read_csv("model_input.csv")
df_points = load_input_data_pts()
teams_all = sorted(df_points["team"].unique())
selected_team = st.selectbox("Select a Team", teams_all)

###########################################
# TAB 1: Points Projections
###########################################
with tabs[0]:
    st.header("Team Points Projections")

    @st.cache_data
    def load_input_data_pts():
        df = pd.read_csv("model_input.csv")
        return df

    @st.cache_resource
    def load_model_pts():
        model = joblib.load("gbr_model_pts.pkl")
        return model

    df_input_pts = load_input_data_pts()
    model_pts = load_model_pts()

    feature_columns_pts = [
        'weighted_pts', 'kalman_pts', 
        'team_proj', 'team_oe', 'team_de', 
        'opp_proj', 'opp_oe', 'opp_de'
    ]
    missing_features_pts = set(feature_columns_pts) - set(df_input_pts.columns)
    if missing_features_pts:
        st.error(f"Missing features in points input data: {missing_features_pts}")
    else:
        df_team_pts = df_input_pts[df_input_pts["team"] == selected_team].copy()
        predictions_pts = model_pts.predict(df_team_pts[feature_columns_pts])
        df_team_pts["predicted_pts"] = predictions_pts

        st.subheader(f"Player Projections for {selected_team} (Points)")
        display_cols_pts = ['player', 'team', 'predicted_pts'] + feature_columns_pts
        st.dataframe(df_team_pts[display_cols_pts])

        st.subheader("Available Betting Lines for Points")
        betting_data = pd.read_csv("odds.csv")
        betting_cols = ['date', 'player_name', 'team_name', 'market_name', 'sportsbook', 'bet_points', 'Over', 'Under']
        bets_points = betting_data[
            (betting_data['team_name'] == selected_team) & 
            (betting_data['market_name'].str.lower() == "player points")
        ]
        if bets_points.empty:
            st.info("No betting lines available for Points for this team.")
        else:
            st.dataframe(bets_points[betting_cols])

        csv_pts = df_team_pts.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Points Projections as CSV",
            data=csv_pts,
            file_name=f"{selected_team}_points_projections.csv",
            mime="text/csv"
        )

###########################################
# TAB 2: Assists Projections
###########################################
with tabs[1]:
    st.header("Team Assists Projections")

    @st.cache_data
    def load_input_data_ast():
        df = pd.read_csv("model_inputs_ast.csv")
        return df

    @st.cache_resource
    def load_model_ast():
        model = joblib.load("gbr_model_ast.pkl")
        return model

    df_input_ast = load_input_data_ast()
    model_ast = load_model_ast()

    feature_columns_ast = [
        'weighted_ast', 'kalman_ast', 
        'team_proj', 'team_oe', 'team_de', 
        'opp_proj', 'opp_oe', 'opp_de'
    ]
    missing_features_ast = set(feature_columns_ast) - set(df_input_ast.columns)
    if missing_features_ast:
        st.error(f"Missing features in assists input data: {missing_features_ast}")
    else:
        df_team_ast = df_input_ast[df_input_ast["team"] == selected_team].copy()
        predictions_ast = model_ast.predict(df_team_ast[feature_columns_ast])
        df_team_ast["predicted_ast"] = predictions_ast

        st.subheader(f"Player Projections for {selected_team} (Assists)")
        display_cols_ast = ['player', 'team', 'predicted_ast'] + feature_columns_ast
        st.dataframe(df_team_ast[display_cols_ast])

        st.subheader("Available Betting Lines for Assists")
        betting_data = pd.read_csv("odds.csv")
        bets_assists = betting_data[
            (betting_data['team_name'] == selected_team) & 
            (betting_data['market_name'].str.lower() == "player assists")
        ]
        if bets_assists.empty:
            st.info("No betting lines available for Assists for this team.")
        else:
            st.dataframe(bets_assists[betting_cols])

        csv_ast = df_team_ast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Assists Projections as CSV",
            data=csv_ast,
            file_name=f"{selected_team}_assists_projections.csv",
            mime="text/csv"
        )

###########################################
# TAB 3: Rebounds Projections
###########################################
with tabs[2]:
    st.header("Team Rebounds Projections")

    @st.cache_data
    def load_input_data_reb():
        df = pd.read_csv("model_inputs_trb.csv")
        return df

    @st.cache_resource
    def load_model_reb():
        model = joblib.load("gbr_model_reb.pkl")
        return model

    df_input_reb = load_input_data_reb()
    model_reb = load_model_reb()

    feature_columns_reb = [
        'weighted_trb', 'kalman_trb', 
        'team_proj', 'team_oe', 'team_de', 
        'opp_proj', 'opp_oe', 'opp_de'
    ]
    missing_features_reb = set(feature_columns_reb) - set(df_input_reb.columns)
    if missing_features_reb:
        st.error(f"Missing features in rebounds input data: {missing_features_reb}")
    else:
        df_team_reb = df_input_reb[df_input_reb["team"] == selected_team].copy()
        predictions_reb = model_reb.predict(df_team_reb[feature_columns_reb])
        df_team_reb["predicted_trb"] = predictions_reb

        st.subheader(f"Player Projections for {selected_team} (Rebounds)")
        display_cols_reb = ['player', 'team', 'predicted_trb'] + feature_columns_reb
        st.dataframe(df_team_reb[display_cols_reb])

        st.subheader("Available Betting Lines for Rebounds")
        betting_data = pd.read_csv("odds.csv")
        bets_reb = betting_data[
            (betting_data['team_name'] == selected_team) & 
            (betting_data['market_name'].str.lower() == "player rebounds")
        ]
        if bets_reb.empty:
            st.info("No betting lines available for Rebounds for this team.")
        else:
            st.dataframe(bets_reb[betting_cols])

        csv_reb = df_team_reb.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Rebounds Projections as CSV",
            data=csv_reb,
            file_name=f"{selected_team}_rebounds_projections.csv",
            mime="text/csv"
        )

###########################################
# TAB 4: Notebook
###########################################
with tabs[3]:
    st.header("Play Notebook")

    st.markdown("Record your plays here. Enter details and click **Add Play** to save a new entry.")

    # Initialize session state to store notebook entries if not already present.
    if "notebook" not in st.session_state:
        st.session_state["notebook"] = []

    # Create a form for adding a new play entry.
    with st.form("notebook_form", clear_on_submit=True):
        entry_date = st.date_input("Date", value=date.today())
        player_name = st.text_input("Player Name")
        team_name = st.text_input("Team")
        book = st.text_input("Book")
        market = st.text_input("Market")
        line = st.text_input("Line")
        odds = st.text_input("Odds")
        projection = st.text_input("Projection")
        submitted = st.form_submit_button("Add Play")

        if submitted:
            new_entry = {
                "date": entry_date.strftime("%Y-%m-%d"),
                "player_name": player_name,
                "team_name": team_name,
                "book": book,
                "market": market,
                "line": line,
                "odds": odds,
                "projection": projection
            }
            st.session_state["notebook"].append(new_entry)
            st.success("Play added successfully!")

    # Convert the notebook entries into a DataFrame for display.
    if st.session_state["notebook"]:
        df_notebook = pd.DataFrame(st.session_state["notebook"])
        st.subheader("Recorded Plays")
        st.dataframe(df_notebook)

        # Optionally, add a download button for the notebook.
        csv_notebook = df_notebook.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Notebook as CSV",
            data=csv_notebook,
            file_name="play_notebook.csv",
            mime="text/csv"
        )
    else:
        st.info("No plays recorded yet.")
