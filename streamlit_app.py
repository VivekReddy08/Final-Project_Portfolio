import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.ERROR)

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
        goals_data = data.groupby(['HomeTeam', 'AwayTeam']).agg({'FTHG': 'sum', 'FTAG': 'sum'}).reset_index()
        pivot_data = goals_data.pivot(index="HomeTeam", columns="AwayTeam", values="FTHG").fillna(0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt="g", cmap="coolwarm")
        plt.title("Heatmap of Goals Scored (Home vs. Away)", color="white")
        st.pyplot(plt)
    except Exception as e:
        logging.error(f"Error in plot_goals_heatmap: {e}")
        st.error(f"Error generating heatmap: {e}")

def plot_avg_goals_trend(data):
    try:
        data['MatchDate'] = pd.to_datetime(data['Date'])
        data.sort_values(by="MatchDate", inplace=True)
        data['TotalGoals'] = data['FTHG'] + data['FTAG']
        goals_trend = data.groupby(data['MatchDate'].dt.to_period("M"))['TotalGoals'].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(goals_trend.index.to_timestamp(), goals_trend.values, marker='o', color='blue')
        plt.title("Average Goals Per Match Over Time", color="white")
        plt.xlabel("Date", color="white")
        plt.ylabel("Average Goals", color="white")
        st.pyplot(plt)
    except Exception as e:
        logging.error(f"Error in plot_avg_goals_trend: {e}")
        st.error(f"Error generating average goals trend: {e}")

def plot_head_to_head_bar(team1, team2, data):
    try:
        h2h_data = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                        ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
        outcomes = h2h_data['FTR'].value_counts()

        outcomes.plot(kind='bar', color=['green', 'yellow', 'red'], figsize=(8, 6))
        plt.title(f"Head-to-Head Results: {team1} vs {team2}", color="white")
        plt.xlabel("Result", color="white")
        plt.ylabel("Count", color="white")
        st.pyplot(plt)
    except Exception as e:
        logging.error(f"Error in plot_head_to_head_bar: {e}")
        st.error(f"Error generating head-to-head bar chart: {e}")

def plot_goal_distribution(data):
    try:
        team_goals = data.groupby("HomeTeam")["FTHG"].sum() + data.groupby("AwayTeam")["FTAG"].sum()
        team_goals = team_goals.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        team_goals.plot(kind="bar", color="orange")
        plt.title("Distribution of Goals by Teams", color="white")
        plt.xlabel("Teams", color="white")
        plt.ylabel("Total Goals", color="white")
        st.pyplot(plt)
    except Exception as e:
        logging.error(f"Error in plot_goal_distribution: {e}")
        st.error(f"Error generating goal distribution: {e}")

def league_prediction(data):
    try:
        st.subheader("League Performance Prediction")
        torres_image = get_base64("steve_torres.jpg")
        st.markdown(
            f"""
            <style>
            .torres-background {{
                background: url(data:image/png;base64,{torres_image});
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 10px;
                margin-top: 10px;
                opacity: 0.9;
            }}
            table {{
                color: white;
                background: rgba(0, 0, 0, 0.8);
                text-align: center;
            }}
            th {{
                color: lightgreen;
                font-size: 14px;
            }}
            td {{
                color: white;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        data['Season'] = pd.to_datetime(data['Date']).dt.year
        latest_season = data['Season'].max()
        season_data = data[data['Season'] == latest_season]

        teams = pd.concat([season_data['HomeTeam'], season_data['AwayTeam']]).unique()

        prediction_results = {}
        for team in teams:
            team_data = season_data[(season_data['HomeTeam'] == team) | (season_data['AwayTeam'] == team)]
            total_matches = len(team_data)
            total_goals_scored = team_data.loc[team_data['HomeTeam'] == team, 'FTHG'].sum() + \
                                 team_data.loc[team_data['AwayTeam'] == team, 'FTAG'].sum()
            total_goals_conceded = team_data.loc[team_data['HomeTeam'] == team, 'FTAG'].sum() + \
                                   team_data.loc[team_data['AwayTeam'] == team, 'FTHG'].sum()

            win_count = len(team_data[((team_data['HomeTeam'] == team) & (team_data['FTR'] == 'H')) |
                                       ((team_data['AwayTeam'] == team) & (team_data['FTR'] == 'A'))])
            draw_count = len(team_data[team_data['FTR'] == 'D'])
            loss_count = total_matches - win_count - draw_count
            avg_goals_scored = total_goals_scored / total_matches
            avg_goals_conceded = total_goals_conceded / total_matches
            win_rate = (win_count / total_matches) * 100
            draw_rate = (draw_count / total_matches) * 100
            loss_rate = (loss_count / total_matches) * 100

            prediction_results[team] = {
                'Avg Goals Scored': round(avg_goals_scored, 2),
                'Avg Goals Conceded': round(avg_goals_conceded, 2),
                'Win Rate (%)': round(win_rate, 2),
                'Draw Rate (%)': round(draw_rate, 2),
                'Loss Rate (%)': round(loss_rate, 2),
            }

        prediction_df = pd.DataFrame(prediction_results).T
        st.markdown(
            f"""
            <div class="torres-background">
                {prediction_df.to_html(index=True, escape=False)}
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        logging.error(f"Error in league_prediction: {e}")
        st.error(f"Error generating league predictions: {e}")

def match_winner_predictor(data):
    try:
        st.subheader("Match Winner Predictor")

        # Team Selection
        team1 = st.selectbox("Select Team 1", data['HomeTeam'].unique(), key="match_team1")
        team2 = st.selectbox("Select Team 2", [t for t in data['AwayTeam'].unique() if t != team1], key="match_team2")

        # Filter data for both teams
        team1_home = data[data['HomeTeam'] == team1]
        team2_away = data[data['AwayTeam'] == team2]

        # Average Goals
        team1_avg_goals = team1_home['FTHG'].mean()
        team2_avg_goals = team2_away['FTAG'].mean()

        # Head-to-Head Stats
        h2h = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                   ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
        team1_wins = len(h2h[h2h['FTR'] == 'H'])
        team2_wins = len(h2h[h2h['FTR'] == 'A'])
        draws = len(h2h[h2h['FTR'] == 'D'])
        total_matches = len(h2h)

        team1_win_prob = (team1_wins / total_matches) * 100 if total_matches > 0 else 0
        team2_win_prob = (team2_wins / total_matches) * 100 if total_matches > 0 else 0
        draw_prob = (draws / total_matches) * 100 if total_matches > 0 else 0

        # Display Insights
        st.write(f"**{team1} Avg Goals (Home):** {team1_avg_goals:.2f}")
        st.write(f"**{team2} Avg Goals (Away):** {team2_avg_goals:.2f}")
        st.write(f"**Head-to-Head (Total Matches):** {total_matches}")
        st.write(f"**{team1} Win %:** {team1_win_prob:.2f}%")
        st.write(f"**{team2} Win %:** {team2_win_prob:.2f}%")
        st.write(f"**Draw %:** {draw_prob:.2f}%")

        # Win Probability Pie Chart
        try:
            labels = [f"{team1} Win", f"{team2} Win", "Draw"]
            probabilities = [team1_win_prob, team2_win_prob, draw_prob]

            plt.figure(figsize=(6, 6))
            plt.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#66b3ff', '#99ff99'])
            plt.title("Win Probability")
            st.pyplot(plt)
        except Exception as e:
            logging.error(f"Error in Win Probability Pie Chart: {e}")
            st.error("Error generating win probability pie chart.")

       
# Improved Goals Distribution Histogram
def plot_goals_distribution(data, team1, team2):
    try:
        h2h = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                   ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
        goals = h2h['FTHG'].tolist() + h2h['FTAG'].tolist()

        plt.figure(figsize=(10, 6))
        n, bins, patches = plt.hist(goals, bins=range(0, max(goals) + 2), color='skyblue', edgecolor='black', alpha=0.8)
        plt.title(f"Goals Distribution in Matches between {team1} and {team2}", fontsize=14)
        plt.xlabel("Number of Goals Scored", fontsize=12)
        plt.ylabel("Frequency of Matches", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Annotating each bar
        for count, x in zip(n, bins[:-1]):
            if count > 0:
                plt.text(x + 0.5, count, f'{int(count)}', ha='center', va='bottom', fontsize=10)

        st.pyplot(plt)
    except Exception as e:
        logging.error(f"Error in Goals Distribution Histogram: {e}")
        st.error("Error generating goals distribution histogram.")

# Improved Head-to-Head Result Bar Chart
def plot_h2h_results_chart(data, team1, team2):
    try:
        h2h = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                   ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]
        team1_wins = len(h2h[h2h['FTR'] == 'H'])
        team2_wins = len(h2h[h2h['FTR'] == 'A'])
        draws = len(h2h[h2h['FTR'] == 'D'])
        results = {"Team 1 Wins": team1_wins, "Team 2 Wins": team2_wins, "Draws": draws}

        plt.figure(figsize=(8, 6))
        bars = plt.bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor="black", alpha=0.9)
        plt.title(f"Head-to-Head Results: {team1} vs {team2}", fontsize=14)
        plt.xlabel("Result", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Annotating each bar
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{int(bar.get_height())}', 
                     ha='center', va='bottom', fontsize=10)

        st.pyplot(plt)
    except Exception as e:
        logging.error(f"Error in Head-to-Head Result Bar Chart: {e}")
        st.error("Error generating head-to-head result bar chart.")

# App Layout with Tabs
def app():
    background_image = get_base64("pl_logo.jpg")
    set_background(background_image)
    st.title("AI-Powered Football Match Outcome Predictor")
    combined_data, filtered_data = load_data()
    model = load_model()

    tab1, tab2, tab3, tab4 = st.tabs(["League Overview", "Team Performance", "Head-to-Head", "Match Prediction"])

    with tab1:
        st.header("League Overview")
        plot_goals_heatmap(combined_data)
        plot_avg_goals_trend(combined_data)
        plot_goal_distribution(combined_data)

    with tab2:
        st.header("Team Performance")
        selected_team = st.selectbox("Select a Team", combined_data['HomeTeam'].unique(), key="team_performance")
        plot_goal_distribution(combined_data, selected_team, selected_team)  # Added single team distribution

    with tab3:
        st.header("Head-to-Head")
        team1 = st.selectbox("Select Team 1", combined_data['HomeTeam'].unique(), key="h2h_team1")
        team2 = st.selectbox("Select Team 2", [t for t in combined_data['AwayTeam'].unique() if t != team1], key="h2h_team2")
        plot_h2h_results_chart(combined_data, team1, team2)

    with tab4:
        st.header("Match Prediction")
        league_prediction(combined_data)
        match_winner_predictor(combined_data)

if __name__ == "__main__":
    app()

