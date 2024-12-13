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

# Set Background Image with Gradient
def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.8)), url(data:image/png;base64,{image_file});
            background-size: cover;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Enhanced Visualizations
def plot_goals_heatmap(data):
    try:
        goals_data = data.groupby(['HomeTeam', 'AwayTeam']).agg({'FTHG': 'sum'}).reset_index()
        pivot_data = goals_data.pivot_table(index="HomeTeam", columns="AwayTeam", values="FTHG", fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt="d", cmap="coolwarm")
        plt.title("Heatmap of Goals Scored (Home vs. Away)", fontsize=16)
        plt.xlabel("Away Team")
        plt.ylabel("Home Team")
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")

def plot_avg_goals_trend(data):
    data['MatchDate'] = pd.to_datetime(data['Date'])
    data.sort_values(by="MatchDate", inplace=True)
    data['TotalGoals'] = data['FTHG'] + data['FTAG']
    goals_trend = data.groupby(data['MatchDate'].dt.to_period("M"))['TotalGoals'].mean()

    plt.figure(figsize=(10, 6))
    goals_trend.plot(kind='line', marker='o', color='blue')
    plt.title("Average Goals Per Match Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Goals")
    st.pyplot(plt)

def plot_possession_trend(team, data):
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    team_data['MatchDate'] = pd.to_datetime(team_data['Date'])
    team_data.sort_values(by="MatchDate", inplace=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x="MatchDate", y="PossessionHome", data=team_data, label="Home Possession")
    sns.lineplot(x="MatchDate", y="PossessionAway", data=team_data, label="Away Possession")
    plt.title(f"Possession Trends: {team}")
    plt.xlabel("Date")
    plt.ylabel("Possession (%)")
    plt.legend()
    st.pyplot(plt)

def plot_head_to_head_bar(team1, team2, data):
    h2h_data = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                    ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
    outcomes = h2h_data['FTR'].value_counts()

    outcomes.plot(kind='bar', color=['green', 'yellow', 'red'], figsize=(8, 6))
    plt.title(f"Head-to-Head Results: {team1} vs {team2}")
    plt.xlabel("Result")
    plt.ylabel("Count")
    st.pyplot(plt)

def league_prediction(data):
    st.subheader("League Performance Prediction")
    teams = pd.concat([data['HomeTeam'], data['AwayTeam']]).unique()

    prediction_results = {}
    for team in teams:
        team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
        avg_goals = team_data['FTHG'].mean() + team_data['FTAG'].mean()
        win_percentage = len(team_data[team_data['FTR'] == 'H']) / len(team_data)
        prediction_results[team] = {
            'Avg Goals': avg_goals,
            'Win %': win_percentage * 100
        }

    prediction_df = pd.DataFrame(prediction_results).T
    st.dataframe(prediction_df.style.highlight_max(axis=0, color="lightgreen"))

# App Layout with Tabs
def app():
    # Load Images
    background_image = get_base64("pl_logo.jpg")
    set_background(background_image)

    st.title("AI-Powered Football Match Outcome Predictor")

    # Load data and model
    combined_data, filtered_data = load_data()
    model = load_model()

    tab1, tab2, tab3, tab4 = st.tabs(["League Overview", "Team Performance", "Head-to-Head", "Match Prediction"])

    with tab1:
        st.header("League Overview")
        plot_goals_heatmap(combined_data)
        plot_avg_goals_trend(combined_data)

    with tab2:
        st.header("Team Performance")
        selected_team = st.selectbox("Select a Team", combined_data['HomeTeam'].unique())
        plot_possession_trend(selected_team, combined_data)

    with tab3:
        st.header("Head-to-Head")
        team1 = st.selectbox("Select Team 1", combined_data['HomeTeam'].unique())
        team2 = st.selectbox("Select Team 2", [t for t in combined_data['AwayTeam'].unique() if t != team1])
        plot_head_to_head_bar(team1, team2, combined_data)

    with tab4:
        st.header("Match Prediction")
        league_prediction(combined_data)

if __name__ == "__main__":
    app()
