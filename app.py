import streamlit as st
import pandas as pd
import joblib

st.title("Team Points Projections")

# Load model input data and the saved model
@st.cache_data
def load_input_data():
    # Load your model input CSV file (which should contain final_columns)
    df = pd.read_csv("model_input.csv")
    return df

@st.cache_resource
def load_model():
    # Load your trained model
    model = joblib.load("gbr_model_pts.pkl")
    return model

df_input = load_input_data()
model = load_model()

# Define the feature columns expected by the model
feature_columns = ['weighted_pts', 'kalman_pts', 
                   'team_proj', 'team_oe', 'team_de', 
                   'opp_proj', 'opp_oe', 'opp_de']

# Check for missing features
missing_features = set(feature_columns) - set(df_input.columns)
if missing_features:
    st.error(f"Missing features in input data: {missing_features}")
else:
    # Create a select box to choose a team
    teams = sorted(df_input["team"].unique())
    selected_team = st.selectbox("Select a Team", teams)

    # Filter the data for the selected team
    df_team = df_input[df_input["team"] == selected_team].copy()

    # Generate predictions for players on the selected team
    predictions = model.predict(df_team[feature_columns])
    df_team["predicted_pts"] = predictions

    # Display the projections for the selected team
    st.subheader(f"Player Projections for {selected_team}")
    display_cols = ['player', 'team', 'predicted_pts'] + feature_columns
    st.dataframe(df_team[display_cols])

    # Optionally, add a download button for the projections
    csv = df_team.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Projections as CSV",
        data=csv,
        file_name=f"{selected_team}_projections.csv",
        mime="text/csv"
    )
