import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Data
@st.cache_data
def load_data():
    combined_data = pd.read_csv("combined_data.csv")  # Replace with your dataset
    filtered_data = pd.read_csv("filtered_data.csv")  # Replace with your dataset
    return combined_data, filtered_data

# Load Model
@st.cache_data
def load_model():
    return joblib.load("ensemble_model.pkl")

# Set Background Image
def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{image_file});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load Image as Base64
import base64

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load Background Image
background_image = get_base64("pl_logo.jpg")

# Interactive Visualizations
def plot_team_performance(team, data):
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    team_data['MatchDate'] = pd.to_datetime(team_data['Date'])
    team_data = team_data.sort_values('MatchDate')

    plt.figure(figsize=(10, 5))
    sns.lineplot(x='MatchDate', y='FTHG', data=team_data, label='Goals Scored')
    sns.lineplot(x='MatchDate', y='FTAG', data=team_data, label='Goals Conceded')
    plt.title(f"Performance Over Time: {team}")
    plt.xlabel("Date")
    plt.ylabel("Goals")
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
    plt.title(f"Goal Distribution for {team}")
    plt.xlabel("Goals")
    plt.ylabel("Frequency")
    st.pyplot(plt)

def plot_match_outcomes(team, data):
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    outcomes = team_data['FTR'].value_counts()
    plt.figure(figsize=(5, 5))
    outcomes.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    plt.title(f"Match Outcomes for {team}")
    st.pyplot(plt)

def plot_head_to_head(team1, team2, data):
    h2h_data = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                    ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
    outcomes = h2h_data['FTR'].value_counts()
    plt.figure(figsize=(5, 5))
    outcomes.plot.pie(autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FFC107', '#F44336'])
    plt.title(f"Head-to-Head: {team1} vs {team2}")
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

    st.write("*Total Matches Played:*", total_matches)
    st.write("*Total Goals Scored:*", total_goals)
    st.write("*Average Goals per Match:*", round(avg_goals_per_match, 2))
    st.write("*Most Common Match Outcome:*", most_wins)

    # Visualization of goals per team
    goals_per_team = data.groupby('HomeTeam')[['FTHG', 'FTAG']].sum()
    goals_per_team['TotalGoals'] = goals_per_team['FTHG'] + goals_per_team['FTAG']
    goals_per_team.sort_values(by='TotalGoals', ascending=False, inplace=True)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=goals_per_team.index, y=goals_per_team['TotalGoals'])
    plt.xticks(rotation=90)
    plt.title("Goals Scored by Teams")
    plt.xlabel("Teams")
    plt.ylabel("Total Goals")
    st.pyplot(plt)

def app():
    # Set Background
    set_background(background_image)

    st.title("AI-Powered Football Match Outcome Predictor")
    st.header("Interactive Analytics and Visualizations")

    # Load data and model
    combined_data, filtered_data = load_data()
    model = load_model()

    # Statistics Panel
    st.sidebar.header("Statistics Panel")
    if st.sidebar.checkbox("Show League Overview"):
        league_overview(combined_data)

    # Team Selection
    teams = pd.concat([combined_data['HomeTeam'], combined_data['AwayTeam']]).unique()
    selected_team = st.selectbox("Select a Team", teams)

    st.subheader("Team Performance Over Time")
    plot_team_performance(selected_team, combined_data)

    st.subheader("Goal Distribution")
    plot_goal_distribution(selected_team, combined_data)

    st.subheader("Match Outcomes")
    plot_match_outcomes(selected_team, combined_data)

    st.subheader("Head-to-Head Comparison")
    team2 = st.selectbox("Select Opponent", [t for t in teams if t != selected_team])
    plot_head_to_head(selected_team, team2, combined_data)

    st.subheader("Team Comparison")
    compare_teams(selected_team, team2, combined_data)

    st.subheader("Predict Outcome")
    HomeGoalAvg = st.number_input("Average Goals by Home Team (Last 5 Matches):", min_value=0.0, step=0.1)
    AwayGoalAvg = st.number_input("Average Goals by Away Team (Last 5 Matches):", min_value=0.0, step=0.1)
    HomeWinRate = st.number_input("Home Team Win Rate:", min_value=0.0, max_value=1.0, step=0.01)
    AwayWinRate = st.number_input("Away Team Win Rate:", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict Outcome"):
        input_data = np.array([[HomeGoalAvg, AwayGoalAvg, HomeWinRate, AwayWinRate]])
        prediction = model.predict(input_data)[0]
        outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        st.write(f"The predicted outcome is: *{outcome_map[prediction]}*")

if _name_ == "_main_":
    app()
