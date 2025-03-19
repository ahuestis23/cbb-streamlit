import streamlit as st
import pandas as pd
import joblib

st.title("Team Projections")

# Create three tabs: Points, Assists, Rebounds
tabs = st.tabs(["Points Projections", "Assists Projections", "Rebounds Projections"])

# Load betting lines data once (cached)
@st.cache_data
def load_betting_data():
    # Ensure betting_lines.csv is in your repo with the proper columns.
    df_bets = pd.read_csv("betting_lines.csv")
    # Optionally, parse the 'date' column as datetime if needed:
    df_bets['date'] = pd.to_datetime(df_bets['date'], errors='coerce')
    return df_bets

betting_data = load_betting_data()
betting_cols = ['date', 'player_name', 'team_name', 'market_name', 'sportsbook', 'bet_points', 'Over', 'Under']

###########################################
# TAB 1: Points Projections
###########################################
with tabs[0]:
    st.header("Team Points Projections")

    @st.cache_data
    def load_input_data_pts():
        # Load your model input CSV file for points projections
        df = pd.read_csv("model_input.csv")
        return df

    @st.cache_resource
    def load_model_pts():
        # Load your trained model for points
        model = joblib.load("gbr_model_pts.pkl")
        return model

    df_input_pts = load_input_data_pts()
    model_pts = load_model_pts()

    # Define the feature columns expected by the points model
    feature_columns_pts = [
        'weighted_pts', 'kalman_pts', 
        'team_proj', 'team_oe', 'team_de', 
        'opp_proj', 'opp_oe', 'opp_de'
    ]

    # Check for missing features
    missing_features_pts = set(feature_columns_pts) - set(df_input_pts.columns)
    if missing_features_pts:
        st.error(f"Missing features in points input data: {missing_features_pts}")
    else:
        # Create a select box to choose a team for points projections
        teams_pts = sorted(df_input_pts["team"].unique())
        selected_team_pts = st.selectbox("Select a Team for Points", teams_pts)

        # Filter the data for the selected team
        df_team_pts = df_input_pts[df_input_pts["team"] == selected_team_pts].copy()

        # Generate predictions for players on the selected team
        predictions_pts = model_pts.predict(df_team_pts[feature_columns_pts])
        df_team_pts["predicted_pts"] = predictions_pts

        # Display the projections for the selected team
        st.subheader(f"Player Projections for {selected_team_pts} (Points)")
        display_cols_pts = ['player', 'team', 'predicted_pts'] + feature_columns_pts
        st.dataframe(df_team_pts[display_cols_pts])

        # Betting Lines for Points
        st.subheader("Available Betting Lines for Points")
        # Filter betting_data for the selected team and market "Points"
        bets_points = betting_data[
            (betting_data['team_name'] == selected_team_pts) & 
            (betting_data['market_name'].str.lower() == "Player Points")
        ]
        if bets_points.empty:
            st.info("No betting lines available for Points for this team.")
        else:
            st.dataframe(bets_points[betting_cols])

        # Optionally, add a download button for the points projections
        csv_pts = df_team_pts.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Points Projections as CSV",
            data=csv_pts,
            file_name=f"{selected_team_pts}_points_projections.csv",
            mime="text/csv"
        )

###########################################
# TAB 2: Assists Projections
###########################################
with tabs[1]:
    st.header("Team Assists Projections")

    @st.cache_data
    def load_input_data_ast():
        # Load your model input CSV file for assists projections
        df = pd.read_csv("model_inputs_ast.csv")
        return df

    @st.cache_resource
    def load_model_ast():
        # Load your trained model for assists
        model = joblib.load("gbr_model_ast.pkl")
        return model

    df_input_ast = load_input_data_ast()
    model_ast = load_model_ast()

    # Define the feature columns expected by the assists model
    feature_columns_ast = [
        'weighted_ast', 'kalman_ast', 
        'team_proj', 'team_oe', 'team_de', 
        'opp_proj', 'opp_oe', 'opp_de'
    ]

    # Check for missing features
    missing_features_ast = set(feature_columns_ast) - set(df_input_ast.columns)
    if missing_features_ast:
        st.error(f"Missing features in assists input data: {missing_features_ast}")
    else:
        # Create a select box to choose a team for assists projections
        teams_ast = sorted(df_input_ast["team"].unique())
        selected_team_ast = st.selectbox("Select a Team for Assists", teams_ast)

        # Filter the data for the selected team
        df_team_ast = df_input_ast[df_input_ast["team"] == selected_team_ast].copy()

        # Generate predictions for players on the selected team for assists
        predictions_ast = model_ast.predict(df_team_ast[feature_columns_ast])
        df_team_ast["predicted_ast"] = predictions_ast

        # Display the projections for the selected team
        st.subheader(f"Player Projections for {selected_team_ast} (Assists)")
        display_cols_ast = ['player', 'team', 'predicted_ast'] + feature_columns_ast
        st.dataframe(df_team_ast[display_cols_ast])

        # Betting Lines for Assists
        st.subheader("Available Betting Lines for Assists")
        bets_assists = betting_data[
            (betting_data['team_name'] == selected_team_ast) & 
            (betting_data['market_name'].str.lower() == "Player Assists")
        ]
        if bets_assists.empty:
            st.info("No betting lines available for Assists for this team.")
        else:
            st.dataframe(bets_assists[betting_cols])

        # Optionally, add a download button for the assists projections
        csv_ast = df_team_ast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Assists Projections as CSV",
            data=csv_ast,
            file_name=f"{selected_team_ast}_assists_projections.csv",
            mime="text/csv"
        )

###########################################
# TAB 3: Rebounds Projections
###########################################
with tabs[2]:
    st.header("Team Rebounds Projections")

    @st.cache_data
    def load_input_data_reb():
        # Load your model input CSV file for rebounds projections
        df = pd.read_csv("model_inputs_trb.csv")
        return df

    @st.cache_resource
    def load_model_reb():
        # Load your trained model for rebounds
        model = joblib.load("gbr_model_reb.pkl")
        return model

    df_input_reb = load_input_data_reb()
    model_reb = load_model_reb()

    # Define the feature columns expected by the rebounds model
    feature_columns_reb = [
        'weighted_trb', 'kalman_trb', 
        'team_proj', 'team_oe', 'team_de', 
        'opp_proj', 'opp_oe', 'opp_de'
    ]

    # Check for missing features
    missing_features_reb = set(feature_columns_reb) - set(df_input_reb.columns)
    if missing_features_reb:
        st.error(f"Missing features in rebounds input data: {missing_features_reb}")
    else:
        # Create a select box to choose a team for rebounds projections
        teams_reb = sorted(df_input_reb["team"].unique())
        selected_team_reb = st.selectbox("Select a Team for Rebounds", teams_reb)

        # Filter the data for the selected team
        df_team_reb = df_input_reb[df_input_reb["team"] == selected_team_reb].copy()

        # Generate predictions for players on the selected team for rebounds
        predictions_reb = model_reb.predict(df_team_reb[feature_columns_reb])
        df_team_reb["predicted_trb"] = predictions_reb

        # Display the projections for the selected team
        st.subheader(f"Player Projections for {selected_team_reb} (Rebounds)")
        display_cols_reb = ['player', 'team', 'predicted_trb'] + feature_columns_reb
        st.dataframe(df_team_reb[display_cols_reb])

        # Betting Lines for Rebounds
        st.subheader("Available Betting Lines for Rebounds")
        bets_reb = betting_data[
            (betting_data['team_name'] == selected_team_reb) & 
            (betting_data['market_name'].str.lower() == "Player Rebounds")
        ]
        if bets_reb.empty:
            st.info("No betting lines available for Rebounds for this team.")
        else:
            st.dataframe(bets_reb[betting_cols])

        # Optionally, add a download button for the rebounds projections
        csv_reb = df_team_reb.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Rebounds Projections as CSV",
            data=csv_reb,
            file_name=f"{selected_team_reb}_rebounds_projections.csv",
            mime="text/csv"
        )
