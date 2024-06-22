import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from scipy.stats import poisson

def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Pre-processing Data Latih", "Generate Data Uji", "Prediksi Pertandingan"],
        icons=["gear", "repeat", "bullseye"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    return selected

def proses_data_latih(df_historical_data, data_range):
    # Display data before preprocessing
    st.write("### Data Latih Sebelum Pre-Processing:")
    st.write(df_historical_data)

    # Bagian pre-processing
    st.write("## Pre-Processing Latih")
    st.write("### Langkah-langkah Pre-Processing Latih:")

    # Clean data
    st.write("- Membersihkan data...")
    df_historical_data['score'] = df_historical_data['score'].str.replace(r'[^\d–]', '', regex=True)
    df_historical_data['home'] = df_historical_data['home'].str.strip()
    df_historical_data['away'] = df_historical_data['away'].str.strip()

    # Split score columns into home and away goals and drop score column
    df_historical_data[['HomeGoals', 'AwayGoals']] = df_historical_data['score'].str.split('–', expand=True)
    df_historical_data.drop('score', axis=1, inplace=True)

    # Rename columns and change data types
    df_historical_data.rename(columns={'home': 'HomeTeam', 'away': 'AwayTeam', 'year': 'Year'}, inplace=True)
    df_historical_data = df_historical_data.astype({'HomeGoals': int, 'AwayGoals': int, 'Year': int})

    # Create new column "TotalGoals"
    df_historical_data['TotalGoals'] = df_historical_data['HomeGoals'] + df_historical_data['AwayGoals']

    # Filter rows based on user choice of data range
    df_historical_data = df_historical_data.iloc[data_range[0]:data_range[1]]

    st.write("### Data setelah proses pre-processing:")
    st.write(df_historical_data)

    # Download button
    st.write("### Download Data Latih yang Telah Diproses:")
    file_name = f"data_latih_{data_range[0]}_{data_range[1]}.csv"
    csv = df_historical_data.to_csv(index=False)
    st.download_button(label="Download CSV File", data=csv, file_name=file_name, mime="text/csv")

    return df_historical_data

selected = streamlit_menu()

if selected == "Pre-processing Data Latih":
    st.info('Anda telah memilih Pre-processing data latih', icon="ℹ️")
    # Load historical data for training from user input
    uploaded_file = st.file_uploader("Upload historical training data (CSV file)", type=["csv"])
    if uploaded_file is not None:
        df_historical_data = pd.read_csv(uploaded_file)

        # Slider for selecting data range
        st.write("### Pilih Range Data Latih:")
        data_range = st.slider("Range Data Latih", 0, len(df_historical_data), (0, len(df_historical_data)))

        # Trigger Pre-Processing Latih Button
        if st.button("Mulai Pre-Processing Latih"):
            proses_data_latih(df_historical_data, data_range)

if selected == "Generate Data Uji":
    st.info('Anda telah memilih Generate data uji', icon="ℹ️")

    # New code for generating testing data
    st.header("Data Uji")
    df = pd.DataFrame(columns=['home', 'away', 'year'])

    # Column configuration for st.data_editor
    config = {
        'home': st.column_config.TextColumn("home", width='large', help=None, disabled=None, required=True, default=None, max_chars=None, validate=None),
        'away': st.column_config.TextColumn("away", width='large', help=None, disabled=None, required=True, default=None, max_chars=None, validate=None),
        'year': st.column_config.TextColumn("year", width='large', help=None, disabled=None, required=True, default=None, max_chars=None, validate=None)
    }

    result = st.data_editor(df, column_config=config, num_rows='dynamic')

    if st.button('Generate Table'):
        st.write(result)

        # Convert DataFrame to CSV and offer download
        result_csv = result.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=result_csv,
            file_name='data_uji.csv',
            mime='text/csv'
        )

# Load historical data for testing from user input
if selected == "Prediksi Pertandingan":
    st.title('Football Prediction Machine')

    # Drag and drop file inputs
    historical_data_file = st.file_uploader('Upload Data Latih', type='csv')
    fixture_file = st.file_uploader('Upload Data Uji', type='csv')

    if historical_data_file is not None and fixture_file is not None:
        df_historical_data = pd.read_csv(historical_data_file)
        df_fixture = pd.read_csv(fixture_file)

        def process_data(df_historical_data, df_fixture):
            df_home = df_historical_data[['HomeTeam', 'HomeGoals', 'AwayGoals']]
            df_away = df_historical_data[['AwayTeam', 'HomeGoals', 'AwayGoals']]

            df_home = df_home.rename(columns={'HomeTeam': 'Team', 'HomeGoals': 'GoalsScored', 'AwayGoals': 'GoalsConceded'})
            df_away = df_away.rename(columns={'AwayTeam': 'Team', 'HomeGoals': 'GoalsConceded', 'AwayGoals': 'GoalsScored'})
            powerlevel = pd.concat([df_home, df_away], ignore_index=True).groupby(['Team']).mean()

            return df_home, df_away, powerlevel

        def calculate_winner(country1_name, country2_name, powerlevel):
            country1_goals_scored = powerlevel.loc[country1_name, 'GoalsScored']
            country1_goals_conceded = powerlevel.loc[country1_name, 'GoalsConceded']

            country2_goals_scored = powerlevel.loc[country2_name, 'GoalsScored']
            country2_goals_conceded = powerlevel.loc[country2_name, 'GoalsConceded']

            lambda_a = country1_goals_scored * country2_goals_conceded
            lambda_b = country2_goals_scored * country1_goals_conceded

            prob_a = prob_b = 0.0

            for x_a in range(7):
                for x_b in range(7):
                    p_total = poisson.pmf(x_a, lambda_a) * poisson.pmf(x_b, lambda_b)
                    if x_a > x_b:
                        prob_a += p_total
                    elif x_a < x_b:
                        prob_b += p_total

            total_prob = prob_a + prob_b
            prob_a /= total_prob
            prob_b /= total_prob

            return prob_a, prob_b

        if st.button("Proses Data"):
            df_home, df_away, powerlevel = process_data(df_historical_data, df_fixture)
            st.write("### Data Home Team:")
            st.write(df_home)
            st.write("### Data Away Team:")
            st.write(df_away)
            st.write("### Data Team Strength:")
            st.write(powerlevel)

            # Determine winners and probabilities for each match in the fixture
            df_fixture['Winner'] = ""
            df_fixture['Home Team Win'] = 0.0
            df_fixture['Away Team Win'] = 0.0

            for index, row in df_fixture.iterrows():
                home, away = row['home'], row['away']
                prob_home_win, prob_away_win = calculate_winner(home, away, powerlevel)
                if prob_home_win > prob_away_win:
                    winner = home
                else:
                    winner = away
                df_fixture.at[index, 'Winner'] = winner
                df_fixture.at[index, 'Home Team Win'] = prob_home_win
                df_fixture.at[index, 'Away Team Win'] = prob_away_win

            st.write("### Updated Fixture:")
            st.write(df_fixture)
