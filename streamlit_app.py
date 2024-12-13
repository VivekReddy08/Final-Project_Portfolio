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
        </style>
        """,
        unsafe_allow_html=True
    )

# Interactive Visualizations
def plot_team_performance(team, data):
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    team_data['MatchDate'] = pd.to_datetime(team_data['Date'])
    team_data = team_data.sort_values('MatchDate')

    plt.figure(figsize=(10, 5))
    sns.lineplot(x='MatchDate', y='FTHG', data=team_data, label='Goals Scored')
    sns.lineplot(x='MatchDate', y='FTAG', data=team_data, label='Goals Conceded')
    plt.title(f"Performance Over Time: {team}", color="white")
    plt.xlabel("Date", color="white")
    plt.ylabel("Goals", color="white")
    plt.legend()
    st.pyplot(plt)

def plot_goal_distribution(team, data):
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    goals = (
        team_data['FTHG'][team_data['HomeTeam'] == team].sum() + 
        team_data['FTAG'][team_data['AwayTeam'] == team].sum()
    )

    plt.figure(figsize=(10, 5))
    sns.histplot([goals], bins=10, kde=True)
    plt.title(f"Goal Distribution for {team}", color="white")
    plt.xlabel("Goals", color="white")
    plt.ylabel("Frequency", color="white")
    st.pyplot(plt)

def plot_match_outcomes(team, data):
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    outcomes = team_data['FTR'].value_counts()
    plt.figure(figsize=(5, 5))
    outcomes.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    plt.title(f"Match Outcomes for {team}", color="white")
    st.pyplot(plt)

def plot_head_to_head(team1, team2, data):
    h2h_data = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                    ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
    outcomes = h2h_data['FTR'].value_counts()
    plt.figure(figsize=(5, 5))
    outcomes.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    plt.title(f"Head-to-Head: {team1} vs {team2}", color="white")
    st.pyplot(plt)

def compare_teams(team1, team2, data):
    team1_data = data[(data['HomeTeam'] == team1) | (data['AwayTeam'] == team1)]
    team2_data = data[(data['HomeTeam'] == team2) | (data['AwayTeam'] == team2)]

    metrics = {
        'Total Matches': [len(team1_data), len(team2_data)],
        'Total Goals Scored': [
            team1_data['FTHG'][team1_data['HomeTeam'] == team1].sum() + team1_data['FTAG'][team1_data['AwayTeam'] == team1].sum(),
            team2_data['FTHG'][team2_data['HomeTeam'] == team2].sum() + team2_data['FTAG'][team2_data['AwayTeam'] == team2].sum()
        ],
        'Total Goals Conceded': [
            team1_data['FTAG'][team1_data['HomeTeam'] == team1].sum() + team1_data['FTHG'][team1_data['AwayTeam'] == team1].sum(),
            team2_data['FTAG'][team2_data['HomeTeam'] == team2].sum() + team2_data['FTHG'][team2_data['AwayTeam'] == team2].sum()
        ]
    }

    comparison_df = pd.DataFrame(metrics, index=[team1, team2])
    st.subheader(f"Team Comparison: {team1} vs {team2}")
    st.table(comparison_df)

# League-Wide Performance Overview
def league_overview(data):
    st.header("League-Wide Performance Overview")

    total_matches = len(data)
    total_goals = data['FTHG'].sum() + data['FTAG'].sum()
    avg_goals_per_match = total_goals / total_matches
    most_wins = data['FTR'].value_counts().idxmax()

    st.write("**Total Matches Played:**", total_matches)
    st.write("**Total Goals Scored:**", total_goals)
    st.write("**Average Goals per Match:**", round(avg_goals_per_match, 2))
    st.write("**Most Common Match Outcome:**", most_wins)

    # Visualization of goals per team
    goals_per_team = data.groupby('HomeTeam')[['FTHG', 'FTAG']].sum()
    goals_per_team['TotalGoals'] = goals_per_team['FTHG'] + goals_per_team['FTAG']
    goals_per_team.sort_values(by='TotalGoals', ascending=False, inplace=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=goals_per_team.index, y=goals_per_team['TotalGoals'])
    plt.xticks(rotation=90)
    plt.title("Goals Scored by Teams", color="white")
    plt.xlabel("Teams", color="white")
    plt.ylabel("Total Goals", color="white")
    st.pyplot(plt)

def app():
    # Load Images
    background_image = get_base64("pl_logo.jpg")  # Replace with your logo
    set_background(background_image)

    st.title("AI-Powered Football Match Outcome Predictor")

    # Load data and model
    combined_data, filtered_data = load_data()
    model = load_model()

    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["League Overview", "Team Performance", "Head-to-Head", "Match Prediction"])

    with tab1:
        league_overview(combined_data)

    with tab2:
        teams = pd.concat([combined_data['HomeTeam'], combined_data['AwayTeam']]).unique()
        selected_team = st.selectbox("Select a Team", teams)
        plot_team_performance(selected_team, combined_data)

    with tab3:
        teams = pd.concat([combined_data['HomeTeam'], combined_data['AwayTeam']]).unique()
        team1 = st.selectbox("Select Team 1", teams)
        team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1])
        plot_head_to_head(team1, team2, combined_data)
        compare_teams(team1, team2, combined_data)

    with tab4:
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

if __name__ == "__main__":
    app()
