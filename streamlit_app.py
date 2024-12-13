import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64

# Load Data
@st.cache_data
def load_data():
    combined_data = pd.read_csv("combined_data.csv")
    filtered_data = pd.read_csv("filtered_data.csv")
    return combined_data, filtered_data

# Load Model
@st.cache_data
def load_model():
    return joblib.load("ensemble_model.pkl")

# Helper Function: Load Base64 Encoded Image
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Set Background Image
def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{image_file});
            background-size: cover;
            color: white;
        }}
        .stMarkdown h1, h2, h3, h4, h5, h6 {{
            color: white;
        }}
        .stMarkdown p {{
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# League-Wide Prediction
def league_prediction(model, combined_data):
    st.header("League-Wide Predictions")
    teams = pd.concat([combined_data['HomeTeam'], combined_data['AwayTeam']]).unique()
    
    # Simulate predictions for every team combination
    predictions = []
    for home_team in teams:
        for away_team in teams:
            if home_team != away_team:
                avg_home_goals = combined_data[combined_data['HomeTeam'] == home_team]['FTHG'].mean()
                avg_away_goals = combined_data[combined_data['AwayTeam'] == away_team]['FTAG'].mean()
                home_win_rate = combined_data[combined_data['HomeTeam'] == home_team]['FTR'].value_counts(normalize=True).get('H', 0)
                away_win_rate = combined_data[combined_data['AwayTeam'] == away_team]['FTR'].value_counts(normalize=True).get('A', 0)
                
                # Create input array
                input_data = np.array([[avg_home_goals, avg_away_goals, home_win_rate, away_win_rate]])
                prediction = model.predict(input_data)[0]
                
                outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
                predictions.append({
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Prediction": outcome_map[prediction]
                })

    # Convert predictions to DataFrame and display
    predictions_df = pd.DataFrame(predictions)
    st.write("**Predicted Match Outcomes for the League**")
    st.dataframe(predictions_df)

    # Visualization of prediction distribution
    prediction_counts = predictions_df['Prediction'].value_counts()
    plt.figure(figsize=(5, 5))
    prediction_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    plt.title("League-Wide Prediction Distribution", color="white")
    st.pyplot(plt)

def app():
    # Load Images
    background_image = get_base64("pl_logo.jpg")  # Replace with your logo
    set_background(background_image)

    st.title("AI-Powered Football Match Outcome Predictor")

    # Load data and model
    combined_data, filtered_data = load_data()
    model = load_model()

    st.sidebar.header("Statistics Panels")

    if st.sidebar.checkbox("League-Wide Overview"):
        league_overview(combined_data)

    teams = pd.concat([combined_data['HomeTeam'], combined_data['AwayTeam']]).unique()

    if st.sidebar.checkbox("Team Performance"):
        selected_team = st.sidebar.selectbox("Select a Team", teams)
        plot_team_performance(selected_team, combined_data)

    if st.sidebar.checkbox("Head-to-Head Comparison"):
        team1 = st.sidebar.selectbox("Select Team 1", teams)
        team2 = st.sidebar.selectbox("Select Team 2", [t for t in teams if t != team1])
        plot_head_to_head(team1, team2, combined_data)
        compare_teams(team1, team2, combined_data)

    if st.sidebar.checkbox("Match Prediction"):
        st.subheader("Predict Outcome")
        HomeGoalAvg = st.number_input("Avg Goals Home Team (Last 5 Matches):", min_value=0.0, step=0.1)
        AwayGoalAvg = st.number_input("Avg Goals Away Team (Last 5 Matches):", min_value=0.0, step=0.1)
        HomeWinRate = st.number_input("Home Win Rate:", min_value=0.0, max_value=1.0, step=0.01)
        AwayWinRate = st.number_input("Away Win Rate:", min_value=0.0, max_value=1.0, step=0.01)

        if st.button("Predict Outcome"):
            input_data = np.array([[HomeGoalAvg, AwayGoalAvg, HomeWinRate, AwayWinRate]])
            prediction = model.predict(input_data)[0]
            outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            st.write(f"The predicted outcome is: **{outcome_map[prediction]}**")

    if st.sidebar.checkbox("League-Wide Prediction"):
        league_prediction(model, combined_data)

if __name__ == "__main__":
    app()
