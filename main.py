import joblib
import pandas as pd

# Load the saved model
rf_model = joblib.load('rf_model.pkl')

# RJ DAVIS test
manual_input = {
    'Min_per': 21,
    'Usage': 0.18,
    'eFG': 0.606,
    'TS_per': 0.6,
    'ORB_per': 0.059,
    'DRB_per': 0.224,
    'AST_per': 0.067,
    'TO_per': 0.152,
    'venue': 0, 
    'team_proj': 73.0, 
    'team_adjt': 70.6,
    'team_adjo': 118.9,
    'team_adjd': 99.8,
    'opp_proj': 71.0,
    'opp_adjt': 65.8,
    'opp_adjo': 108.9,
    'opp_adjd': 94.3
}

# Convert dictionary to DataFrame
manual_df = pd.DataFrame([manual_input]).astype(float)

# Make prediction
predicted_points = rf_model.predict(manual_df)[0]

print(f"Predicted Points: {predicted_points:.2f}")
