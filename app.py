import streamlit as st
import pandas as pd
import joblib

st.title("Team Projections")

# Create two tabs: one for Points and one for Assists
tabs = st.tabs(["Points Projections", "Assists Projections"])

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

        # Optionally, add a download button for the assists projections
        csv_ast = df_team_ast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Assists Projections as CSV",
            data=csv_ast,
            file_name=f"{selected_team_ast}_assists_projections.csv",
            mime="text/csv"
        )
###########################################
# TAB 4: Rebounds Projections
###########################################
with tabs[1]:
    st.header("Team Rebounds Projections")

    @st.cache_data
    def load_input_data_ast():
        # Load your model input CSV file for assists projections
        df = pd.read_csv("model_inputs_trb.csv")
        return df

    @st.cache_resource
    def load_model_ast():
        # Load your trained model for assists
        model = joblib.load("gbr_model_reb.pkl")
        return model

    df_input_ast = load_input_data_ast()
    model_ast = load_model_ast()

    # Define the feature columns expected by the assists model
    feature_columns_ast = [
        'weighted_trb', 'kalman_trb', 
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
        selected_team_ast = st.selectbox("Select a Team for Rebounds", teams_ast)

        # Filter the data for the selected team
        df_team_ast = df_input_ast[df_input_ast["team"] == selected_team_ast].copy()

        # Generate predictions for players on the selected team for assists
        predictions_ast = model_ast.predict(df_team_ast[feature_columns_ast])
        df_team_ast["predicted_trb"] = predictions_ast

        # Display the projections for the selected team
        st.subheader(f"Player Projections for {selected_team_ast} (Rebounds)")
        display_cols_ast = ['player', 'team', 'predicted_trb'] + feature_columns_ast
        st.dataframe(df_team_ast[display_cols_ast])

        # Optionally, add a download button for the assists projections
        csv_ast = df_team_ast.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Rebounds Projections as CSV",
            data=csv_ast,
            file_name=f"{selected_team_ast}_assists_projections.csv",
            mime="text/csv"
        )
