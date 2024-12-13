import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Helper Functions
def league_overview(data):
    st.header("League Overview")
    
    st.subheader("Total Goals Scored Per Team")
    goals_per_team = data.groupby('HomeTeam')['FTHG'].sum() + data.groupby('AwayTeam')['FTAG'].sum()
    goals_per_team.sort_values(ascending=False, inplace=True)
    st.bar_chart(goals_per_team)

    st.subheader("Match Outcome Distribution")
    outcome_distribution = data['FTR'].value_counts()
    fig, ax = plt.subplots()
    outcome_distribution.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FFC107', '#F44336'])
    ax.set_ylabel('')
    st.pyplot(fig)

def team_performance(data, team):
    st.header(f"Performance: {team}")
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]
    team_data['MatchDate'] = pd.to_datetime(team_data['Date'])
    team_data = team_data.sort_values('MatchDate')

    st.subheader("Goals Scored vs. Goals Conceded Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(x='MatchDate', y='FTHG', data=team_data, ax=ax, label='Goals Scored')
    sns.lineplot(x='MatchDate', y='FTAG', data=team_data, ax=ax, label='Goals Conceded')
    ax.set_ylabel('Goals')
    st.pyplot(fig)

    st.subheader("Match Outcomes")
    outcomes = team_data['FTR'].value_counts()
    fig, ax = plt.subplots()
    outcomes.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

def head_to_head(data, team1, team2):
    st.header(f"Head-to-Head: {team1} vs {team2}")
    h2h_data = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                    ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]

    st.subheader("Win/Draw/Loss Distribution")
    outcomes = h2h_data['FTR'].value_counts()
    fig, ax = plt.subplots()
    outcomes.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
    ax.set_ylabel('')
    st.pyplot(fig)

    st.subheader("Goals Comparison")
    goals = {
        team1: h2h_data['FTHG'][h2h_data['HomeTeam'] == team1].sum() + h2h_data['FTAG'][h2h_data['AwayTeam'] == team1].sum(),
        team2: h2h_data['FTHG'][h2h_data['HomeTeam'] == team2].sum() + h2h_data['FTAG'][h2h_data['AwayTeam'] == team2].sum()
    }
    fig, ax = plt.subplots()
    ax.bar(goals.keys(), goals.values(), color=['blue', 'red'])
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
