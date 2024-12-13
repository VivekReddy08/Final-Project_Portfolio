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
    
    st.subheader("Total Goals Scored Per Team")
    goals_per_team = data.groupby('HomeTeam')['FTHG'].sum() + data.groupby('AwayTeam')['FTAG'].sum()
    goals_per_team.sort_values(ascending=False, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=goals_per_team.index, y=goals_per_team.values, ax=ax, palette="coolwarm")
    ax.set_xticklabels(goals_per_team.index, rotation=45)
    ax.set_title("Goals Scored by Teams")
    ax.set_ylabel("Total Goals")
    st.pyplot(fig)

    st.subheader("Win/Draw/Loss Distribution (League-wide)")
    outcome_distribution = data['FTR'].value_counts()
    fig, ax = plt.subplots()
    outcome_distribution.plot.pie(
        autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FFC107', '#F44336']
    )
    ax.set_ylabel('')
    ax.set_title("League-Wide Match Outcomes")
    st.pyplot(fig)

    st.subheader("Top Scoring Matches")
    data['TotalGoals'] = data['FTHG'] + data['FTAG']
    top_matches = data.nlargest(10, 'TotalGoals')[['Date', 'HomeTeam', 'AwayTeam', 'TotalGoals']]
    st.table(top_matches)

def team_performance(data, team):
    st.header(f"Performance: {team}")

    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    team_data['MatchDate'] = pd.to_datetime(team_data['Date'])
    team_data = team_data.sort_values('MatchDate')

    st.subheader("Goals Scored vs. Goals Conceded Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        x='MatchDate', y='FTHG', data=team_data, label='Goals Scored', ax=ax
    )
    sns.lineplot(
        x='MatchDate', y='FTAG', data=team_data, label='Goals Conceded', ax=ax
    )
    ax.set_title("Performance Over Time")
    ax.set_ylabel("Goals")
    st.pyplot(fig)

    st.subheader("Match Outcomes")
    outcomes = team_data['FTR'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    outcomes.plot.pie(
        autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FFC107', '#F44336']
    )
    ax.set_ylabel('')
    ax.set_title("Match Outcomes")
    st.pyplot(fig)

    st.subheader("Goals Scored in Last 5 Matches")
    last_5_matches = team_data.tail(5)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=last_5_matches['MatchDate'].dt.strftime('%Y-%m-%d'),
        y=last_5_matches['FTHG'] + last_5_matches['FTAG'],
        ax=ax,
        palette="viridis",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Goals Scored in Last 5 Matches")
    ax.set_ylabel("Goals")
    st.pyplot(fig)

def head_to_head(data, team1, team2):
    st.header(f"Head-to-Head: {team1} vs {team2}")

    h2h_data = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                    ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]

    st.subheader("Win/Draw/Loss Distribution")
    outcomes = h2h_data['FTR'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    outcomes.plot.pie(
        autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FFC107', '#F44336']
    )
    ax.set_ylabel('')
    ax.set_title("Head-to-Head Match Outcomes")
    st.pyplot(fig)

    st.subheader("Goals Comparison")
    goals = {
        team1: h2h_data['FTHG'][h2h_data['HomeTeam'] == team1].sum() + h2h_data['FTAG'][h2h_data['AwayTeam'] == team1].sum(),
        team2: h2h_data['FTHG'][h2h_data['HomeTeam'] == team2].sum() + h2h_data['FTAG'][h2h_data['AwayTeam'] == team2].sum()
    }
    fig, ax = plt.subplots()
    sns.barplot(x=list(goals.keys()), y=list(goals.values()), ax=ax, palette="coolwarm")
    ax.set_title("Goals Scored")
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
    # Add background
    background_image = get_base64("pl_logo.jpg")  # Replace with your background image
    set_background(background_image)

    st.title("AI-Powered Football Match Outcome Predictor")
    tab1, tab2, tab3, tab4 = st.tabs(["League Overview", "Team Performance", "Head-to-Head", "Match Prediction"])

    with tab1:
        league_overview(data)

    with tab2:
        team = st.selectbox("Select a Team", data['HomeTeam'].unique())
        team_performance(data, team)

    with tab3:
        team1 = st.selectbox("Select Team 1", data['HomeTeam'].unique(), key="team1")
        team2 = st.selectbox("Select Team 2", [t for t in data['HomeTeam'].unique() if t != team1], key="team2")
        head_to_head(data, team1, team2)

    with tab4:
        match_prediction()

if __name__ == "__main__":
    data = pd.read_csv("combined_data.csv")
    app(data)
