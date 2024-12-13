import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

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
        .gradient-box {{
            background: linear-gradient(to bottom right, rgba(0,0,0,0.8), rgba(0,0,0,0.5));
            padding: 10px;
            border-radius: 10px;
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

# League Overview
def league_overview(data):
    st.header("League Overview")
    st.subheader("Top Scoring Matches")
    
    top_matches = data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
    top_matches['TotalGoals'] = top_matches['FTHG'] + top_matches['FTAG']
    top_matches = top_matches.sort_values(by='TotalGoals', ascending=False).head(10)

    st.markdown('<div class="gradient-box">', unsafe_allow_html=True)
    st.table(top_matches[['Date', 'HomeTeam', 'AwayTeam', 'TotalGoals']])
    st.markdown('</div>', unsafe_allow_html=True)

    # Visualization: Goals Scored by Teams
    st.subheader("Goals Scored by Teams")
    team_goals = data.groupby('HomeTeam')[['FTHG']].sum().reset_index()
    team_goals.columns = ['Team', 'GoalsScored']
    team_goals = team_goals.sort_values(by='GoalsScored', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Team', y='GoalsScored', data=team_goals, ax=ax, palette='coolwarm')
    ax.set_title("Total Goals Scored by Teams", fontsize=16, color="white")
    ax.set_xlabel("Teams", fontsize=12, color="white")
    ax.set_ylabel("Goals Scored", fontsize=12, color="white")
    ax.tick_params(colors='white')
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Team Performance
def team_performance(data, team):
    st.header(f"Performance of {team}")
    team_data = data[(data['HomeTeam'] == team) | (data['AwayTeam'] == team)]

    # Goals Over Time
    team_data['MatchDate'] = pd.to_datetime(team_data['Date'])
    team_data = team_data.sort_values('MatchDate')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='MatchDate', y='FTHG', data=team_data[team_data['HomeTeam'] == team], label='Goals Scored', ax=ax, color='blue')
    sns.lineplot(x='MatchDate', y='FTAG', data=team_data[team_data['AwayTeam'] == team], label='Goals Conceded', ax=ax, color='red')
    ax.set_title(f"Goals Over Time for {team}", fontsize=14, color="white")
    ax.set_ylabel("Goals", fontsize=12, color="white")
    ax.set_xlabel("Match Date", fontsize=12, color="white")
    ax.tick_params(colors='white')
    st.pyplot(fig)

# Head-to-Head Comparison
def head_to_head_comparison(data, team1, team2):
    st.header(f"Head-to-Head: {team1} vs {team2}")

    # Filter data for head-to-head matches
    h2h_data = data[((data['HomeTeam'] == team1) & (data['AwayTeam'] == team2)) |
                    ((data['HomeTeam'] == team2) & (data['AwayTeam'] == team1))]

    if h2h_data.empty:
        st.markdown('<div class="gradient-box">', unsafe_allow_html=True)
        st.write(f"No matches found between **{team1}** and **{team2}**.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display statistics table
        st.subheader("Head-to-Head Statistics")
        total_matches = len(h2h_data)
        team1_wins = len(h2h_data[(h2h_data['HomeTeam'] == team1) & (h2h_data['FTR'] == 'H')]) + \
                     len(h2h_data[(h2h_data['AwayTeam'] == team1) & (h2h_data['FTR'] == 'A')])
        team2_wins = len(h2h_data[(h2h_data['HomeTeam'] == team2) & (h2h_data['FTR'] == 'H')]) + \
                     len(h2h_data[(h2h_data['AwayTeam'] == team2) & (h2h_data['FTR'] == 'A')])
        draws = len(h2h_data[h2h_data['FTR'] == 'D'])

        stats = {
            "Total Matches": total_matches,
            f"Wins by {team1}": team1_wins,
            f"Wins by {team2}": team2_wins,
            "Draws": draws,
            f"Total Goals by {team1}": h2h_data[h2h_data['HomeTeam'] == team1]['FTHG'].sum() + h2h_data[h2h_data['AwayTeam'] == team1]['FTAG'].sum(),
            f"Total Goals by {team2}": h2h_data[h2h_data['HomeTeam'] == team2]['FTHG'].sum() + h2h_data[h2h_data['AwayTeam'] == team2]['FTAG'].sum()
        }
        stats_df = pd.DataFrame(stats.items(), columns=["Metric", "Value"])
        st.markdown('<div class="gradient-box">', unsafe_allow_html=True)
        st.table(stats_df)
        st.markdown('</div>', unsafe_allow_html=True)

        # Display outcome pie chart
        st.subheader("Match Outcomes")
        outcomes = h2h_data['FTR'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        outcomes.plot.pie(
            autopct='%1.1f%%', startangle=90, ax=ax, colors=['#4CAF50', '#FFC107', '#F44336']
        )
        ax.set_ylabel('')
        ax.set_title("Head-to-Head Outcomes", fontsize=14, color="white")
        st.pyplot(fig)

# Match Prediction Placeholder
def match_prediction():
    st.header("Match Prediction")
    st.write("Feature coming soon!")

# Main App
def app(data):
    pl_logo = get_base64("pl_logo.jpg")
    set_background(pl_logo)

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
        head_to_head_comparison(data, team1, team2)

    with tab4:
        match_prediction()

if __name__ == "__main__":
    data = pd.read_csv("combined_data.csv")
    app(data)
