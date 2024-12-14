# AI-Powered Football Match Outcome Predictor

## Project Overview
This project is an AI-powered football match outcome predictor that leverages advanced data analytics and interactive visualizations to provide insights into English Premier League matches. Built using Python and Streamlit, the application offers users an engaging and intuitive interface to explore football data, visualize trends, analyze team and player performance, and predict match outcomes.

## Project Objectives
- **Enhance football analytics**: Provide users with a comprehensive platform to explore match data, team performance, and player insights.
- **Predict outcomes**: Use machine learning models to predict match results.
- **Interactive Visualizations**: Offer users intuitive, visually appealing charts and insights.

## Key Features
1. **League Overview Tab**:
   - Displays an overview of the English Premier League.
   - Provides visualizations of goal distributions, trends, and heatmaps of match data.
   - Includes a brief background on the league.

2. **Team Performance Tab**:
   - Analyze team performance with detailed metrics like wins, losses, and draws.
   - View player analytics, including goals, assists, and matches played.
   - Visualize performance using bar charts and tables.

3. **Head-to-Head Tab**:
   - Compare the performance of two teams head-to-head.
   - Visualize win probabilities, average goals, and match outcomes with interactive charts and pie charts.
   - New: Additional visualizations for head-to-head comparisons.

4. **Match Prediction Tab**:
   - Predict match outcomes using machine learning models.
   - Provide insights into win probabilities, goals scored, and clean sheets.
   - Suggest the likely winner or indicate a draw.

## Technologies Used
- **Frontend**: Streamlit for interactive and user-friendly UI.
- **Data Handling**: Pandas and NumPy for data preprocessing and manipulation.
- **Visualizations**: Matplotlib and Seaborn for charts and graphs.
- **Machine Learning**: Joblib for model loading and prediction.

## Data Details
The application uses match data from the English Premier League, including:
- Match dates, teams, and results.
- Goals scored and conceded by teams.
- Player-level data such as goals, assists, and matches played.

### Data Files
1. `combined_data.csv`: Contains match-level data for analysis.
2. `filtered_data.csv`: Preprocessed data for predictions and visualizations.

## How to Run the Application
1. **Clone the Repository**:
   ```bash
   git clone <repository_link>
   cd <repository_folder>
   ```
2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed. Then, run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the App**:
   ```bash
   streamlit run streamlit_app.py
   ```
4. **Access the App**:
   Open the link provided in the terminal (usually `http://localhost:8501`).

## Application Hosting
The application is hosted on Streamlit Cloud. Access it here:
[Streamlit App Link](#)

## Video Walkthrough
A detailed video walkthrough of the project can be found here:
[YouTube Video Link](#)

## Challenges and Lessons Learned
- **Data Cleaning**: Managing missing or inconsistent data in the match dataset required additional preprocessing steps.
- **Model Integration**: Ensuring seamless integration between the machine learning model and the Streamlit app.
- **UI Enhancements**: Designing an intuitive and visually appealing interface with interactive tabs and visualizations.

## Future Improvements
- **Expand Data Coverage**: Include additional leagues and historical data for broader insights.
- **Advanced Predictions**: Enhance the machine learning model with additional features and fine-tuning.
- **Player Insights**: Incorporate advanced player analytics using additional datasets.

## Repository Structure
```
|-- README.md
|-- app/
|   |-- streamlit_app.py
|   |-- combined_data.csv
|   |-- filtered_data.csv
|-- model/
|   |-- ensemble_model.pkl
|-- requirements.txt
```

## Contact Information
For questions or feedback, please contact:
- **Name**: Vivekavardhan Reddy Alla, Manish Reddy

---

This project was developed as part of the Final Project for the Generative AI course. It demonstrates the integration of AI and data analytics into a practical, user-friendly application.

