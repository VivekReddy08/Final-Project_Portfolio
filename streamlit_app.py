import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Helper Functions
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{image_file});
            background-size: cover;
            color: white;
        }}
        table {{
            color: white !important;
        }}
        th, td {{
            color: white !important;
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

# Visualization Functions
def league_overview(data):
    st.header("League Overview")
    
    # Top Scoring Matches
    st.subheader("Top Scoring Matches")
    data['TotalGoals'] = data['FTHG'] + data['FTAG']
    top_matches = data.nlargest(10, 'TotalGoals')[['Date', 'HomeTeam', 'AwayTeam', 'TotalGoals']]
    st.table(top_matches)

    # Goals Scored Per Team
    st.subheader("Total Goals Scored Per Team")
    goals_per_team = data.groupby('HomeTeam')['FTHG'].sum() + data.groupby('AwayTeam')['FTAG'].sum()
    goals_per_team.sort_values(ascending=False, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=goals_per_team.index, y=goals_per_team.values, ax=ax, palette="coolwarm")
    ax.set_xticklabels(goals_per_team.index, rotation=45)
    ax.set_title("Goals Scored by Teams", fontsize=14, color="white")
    ax.set_ylabel("Total Goals", fontsize=12, color="white")
    ax.set_xlabel("Teams", fontsize=12, color="white")
    st.pyplot(fig)

    # Match Outcome Distribution
    st.subheader("Win/Draw/Loss Distribution (League-wide)")
    outcome_distribution = data['FTR'].value_counts()
    fig, ax = plt.subplots()
    outcome_distribution.plot.pie(
        autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FFC107', '#F44336']
    )
    ax.set_ylabel('')
    ax.set_title("League-Wide Match Outcomes", fontsize=14, color="white")
    st.pyplot(fig)

def team_performance(data, team):
    st.header(f"Performance: {team}")

    # Goals Over Time
    st.subheader("Goals Scored vs. Goals Conceded Over Time")
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    team_data['MatchDate'] = pd.to_datetime(team_data['Date'])
    team_data = team_data.sort_values('MatchDate')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        x='MatchDate', y='FTHG', data=team_data, label='Goals Scored', ax=ax, color="cyan"
    )
    sns.lineplot(
        x='MatchDate', y='FTAG', data=team_data, label='Goals Conceded', ax=ax, color="orange"
    )
    ax.set_title("Performance Over Time", fontsize=14, color="white")
    ax.set_ylabel("Goals", fontsize=12, color="white")
    ax.set_xlabel("Match Date", fontsize=12, color="white")
    st.pyplot(fig)

    # Match Outcomes
    st.subheader("Match Outcomes")
    outcomes = team_data['FTR'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    outcomes.plot.pie(
        autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FFC107', '#F44336']
    )
    ax.set_ylabel('')
    ax.set_title("Match Outcomes", fontsize=14, color="white")
    st.pyplot(fig)

def match_prediction():
    st.header("Match Prediction")
    st.write("Provide inputs for match prediction.")
    HomeGoalAvg = st.number_input("Avg Goals Home Team (Last 5 Matches):", min_value=0.0, step=0.1)
    AwayGoalAvg = st.number_input("Avg Goals Away Team (Last 5 Matches):", min_value=0.0, step=0.1)
    HomeWinRate = st.number_input("Home Win Rate:", min_value=0.0, max_value=1.0, step=0.01)
    AwayWinRate = st.number_input("Away Win Rate:", min_value=0.0, max_value=1.0, step=0.01)

    if st.button("Predict Outcome"):
        prediction = np.random.choice(["Home Win", "Draw", "Away Win"])  # Placeholder prediction logic
        st.write(f"Predicted Outcome: **{prediction}**")

# Main Application
def app(data):
    pl_logo = get_base64("pl_logo.jpg")
    salah_image = get_base64("mo_salah.jpg")
    torres_image = get_base64("steve_torres.jpg")

    set_background(pl_logo)

    st.title("AI-Powered Football Match Outcome Predictor")
    tab1, tab2, tab3, tab4 = st.tabs(["League Overview", "Team Performance", "Head-to-Head", "Match Prediction"])

    with tab1:
        league_overview(data)

    with tab2:
        team = st.selectbox("Select a Team", data['HomeTeam'].unique())
        set_background(salah_image)
        team_performance(data, team)

    with tab3:
        set_background(pl_logo)
        team1 = st.selectbox("Select Team 1", data['HomeTeam'].unique(), key="team1")
        team2 = st.selectbox("Select Team 2", [t for t in data['HomeTeam'].unique() if t != team1], key="team2")
        # Add Head-to-Head Comparison Logic Here

    with tab4:
        set_background(torres_image)
        match_prediction()

if __name__ == "__main__":
    data = pd.read_csv("combined_data.csv")
    app(data)
