import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from datetime import datetime
import pandas as pd
import sqlite3
import requests
import random
from statsmodels.tsa.arima.model import ARIMA  # Import ARIMA for time-series prediction



def get_daily_quote():
    try:
        # ZenQuotes API URL
        response = requests.get("https://zenquotes.io/api/random")
        if response.status_code == 200:
            data = response.json()
            return data[0]['q'], data[0]['a']
        else:
            return "Could not fetch quote", "API Error"
    except requests.exceptions.RequestException as e:
        return f"Error fetching quote: {e}", ""



# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


# SQLite Database setup
def init_db():
    conn = sqlite3.connect('mood_tracker.db')
    c = conn.cursor()
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS mood_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            text TEXT,
            sentiment TEXT,
            score REAL,
            recommendation TEXT
        )
    ''')
    conn.commit()
    conn.close()


# Insert mood entry into the database
def insert_mood_entry(date, text, sentiment, score, recommendation):
    conn = sqlite3.connect('mood_tracker.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO mood_history (date, text, sentiment, score, recommendation)
        VALUES (?, ?, ?, ?, ?)
    ''', (date, text, sentiment, score, recommendation))
    conn.commit()
    conn.close()


# Retrieve mood history from the database
def get_mood_history():
    conn = sqlite3.connect('mood_tracker.db')
    c = conn.cursor()
    c.execute('SELECT date, text, sentiment, score, recommendation FROM mood_history')
    data = c.fetchall()
    conn.close()
    return data


# Initialize the database
init_db()


# Function for speech-to-text conversion
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your input...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            st.session_state["user_text"] = text  # Store in session state
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand.")
        except sr.RequestError:
            st.error("There was an error with the speech recognition service.")


# Function to analyze sentiment of the text with adjusted score and emoji
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    sentiment = result['label']
    score = result['score']

    # Check for mixed emotions (e.g., "happy but tired")
    mixed_emotions = ["but tired", "but exhausted", "but stressed", "but overwhelmed"]
    if any(phrase in text.lower() for phrase in mixed_emotions):
        sentiment = "MIXED"
        adjusted_score = 0  # Reset score to neutral for mixed emotions

    # Handle neutral sentiment based on score threshold
    elif sentiment == "NEUTRAL" or abs(score - 0.5) < 0.1:
        sentiment = "NEUTRAL"
        adjusted_score = 0
    else:
        # Adjust score based on sentiment
        adjusted_score = score if sentiment == "POSITIVE" else -score

    # Assign emoji based on sentiment
    emoji = "😊" if sentiment == "POSITIVE" else "😞" if sentiment == "NEGATIVE" else "😐" if sentiment == "NEUTRAL" else "🤔"

    st.write(f"Sentiment: {emoji} {sentiment}, Adjusted Score: {adjusted_score}")
    return sentiment, adjusted_score, emoji


# Function to generate recommendations with emojis
def get_recommendation(sentiment, score):
    if sentiment == "POSITIVE":
        if score > 0.7:
            return "😊 You're feeling positive! Keep this momentum by doing something productive—maybe review your notes or organize for the upcoming week."
        else:
            return "😊 You seem to be in a good mood. A short walk or a fun hobby break might help keep up the energy for studies!"
    elif sentiment == "NEGATIVE":
        if score > 0.7:
            return "😞 It sounds like you’re feeling quite stressed. Take a break—stretch, breathe, or chat with a friend. You’re not alone in this."
        else:
            return "😞 You might be feeling a bit down. Maybe try a short relaxation exercise, or spend a few minutes on something you enjoy to reset."
    elif sentiment == "MIXED":
        return "🤔 It seems you're experiencing a mix of emotions. Reflect on your thoughts, take some time for yourself, and balance your day with activities that soothe you."
    else:
        return "😐 I'm here to support you. Short acts of self-care, like jotting down your thoughts or doing a 5-minute mindfulness exercise, could boost your mood."


# Function to clear mood history from the database
def clear_mood_history():
    conn = sqlite3.connect('mood_tracker.db')
    c = conn.cursor()
    c.execute('DELETE FROM mood_history')
    conn.commit()
    conn.close()


# Function to clear recommendation history from the database
def clear_recommendation_history():
    conn = sqlite3.connect('mood_tracker.db')
    c = conn.cursor()
    # Delete all records from the mood_history table
    c.execute('DELETE FROM mood_history')
    conn.commit()
    conn.close()


# Main Streamlit app
def main():

    st.set_page_config(page_title="Mental Well-being Application", layout="wide")
    # Custom CSS for a colorful layout
    st.markdown("""
        <style>
        .main {
            background-color: #f0f8ff;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stTextInput>label {
            color: #4CAF50;
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            border-color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)
    st.title("Mental Well-being Application for Students")
    st.write("Analyze your mood and receive tailored recommendations.")

    # Fetch daily quote
    quote, author = get_daily_quote()

    # Display the daily quote at the top of the app
    st.markdown("### Daily Quote")
    st.write(f"**\"{quote}\"**")
    st.write(f"- {author}")

    # Information about the best times to use the app
    st.markdown("### Suggested Times for Using This App")
    st.write("""
    **Morning (Starting the Day)**
    A morning check-in can help you set a positive tone. Assess your mood and set goals for the day.

    **Afternoon (Midday Check-In)**
    Manage midday stress with a check-in to ensure you’re balancing productivity with well-being.

    **Evening (Reflecting on the Day)**
    Reflect on your day’s achievements and challenges, preparing for tomorrow with clarity.

    **Night (Pre-Sleep Routine)**
    Calm your mind and prepare for a restful night by processing thoughts and focusing on self-care.
    """)

    # Tabs for easy navigation
    tabs = st.tabs(["Mood Analysis", "Mood Tracker", "Recommendation History", "Resources"])

    # Mood Analysis Tab
    with tabs[0]:
        st.subheader("Analyze Your Mood")

        # Check if session state has user_text, or initialize
        if "user_text" not in st.session_state:
            st.session_state["user_text"] = ""

        # Display text input field with session state value and unique key
        user_text = st.text_input("Type your thoughts here:", value=st.session_state["user_text"], key="user_text_input")

        # Button to trigger speech recognition
        if st.button("Speak"):
            st.write("Please wait, listening...")
            speech_to_text()  # This will update st.session_state["user_text"]

        # Button to submit text input for sentiment analysis
        if st.button("Submit for Analysis") and user_text:
            # Use session state user_text for analysis if available
            sentiment, adjusted_score, emoji = analyze_sentiment(user_text)
            recommendation = get_recommendation(sentiment, abs(adjusted_score))
            st.write("Recommendation:", recommendation)

            # Save to SQLite database for mood tracking
            insert_mood_entry(datetime.now(), user_text, sentiment, adjusted_score, recommendation)
            st.success("Your mood entry and recommendation have been saved.")



    # Mood Tracker Tab
    with tabs[1]:
        st.subheader("Mood Tracker")
        history_data = get_mood_history()
        if history_data:
            # Create DataFrame with adjusted scores and emojis
            history_df = pd.DataFrame(history_data, columns=["Date", "Text", "Sentiment", "Score", "Recommendation"])
            history_df["Emoji"] = history_df["Sentiment"].apply(
                lambda x: "😊" if x == "POSITIVE" else "😞" if x == "NEGATIVE" else "😐" if x == "NEUTRAL" else "🤔")

            # Convert the 'Date' column to datetime for proper sorting and plotting
            history_df['Date'] = pd.to_datetime(history_df['Date'])
            history_df.set_index('Date', inplace=True)

            # Plot mood over time using Adjusted Score
            mood_chart = history_df[['Score']].copy()  # Use 'Score' or 'Adjusted Score'
            mood_chart['Adjusted Score'] = history_df.apply(
                lambda row: row['Score'] if row['Sentiment'] == "POSITIVE" else -row['Score'], axis=1)

            st.line_chart(mood_chart["Adjusted Score"], use_container_width=True)

            # Add Clear History button
            if st.button("Clear History"):
                clear_mood_history()
                st.success("Mood history has been cleared.")
        else:
            st.write("No mood data available yet.")

    # Recommendation History Tab
    with tabs[2]:
        st.subheader("Recommendation History")
        if history_data:
            history_df = pd.DataFrame(history_data, columns=["Date", "Text", "Sentiment", "Score", "Recommendation"])
            history_df["Emoji"] = history_df["Sentiment"].apply(
                lambda x: "😊" if x == "POSITIVE" else "😞" if x == "NEGATIVE" else "😐" if x == "NEUTRAL" else "🤔")
            st.table(history_df[["Date", "Text", "Sentiment", "Emoji", "Recommendation"]])
        else:
            st.write("No recommendations yet.")

        # Button to clear recommendation history
        if st.button("Clear Recommendation History"):
            clear_recommendation_history()
            st.success("Recommendation history has been cleared.")

    # Resources Tab
    with tabs[3]:

        # Define wellness resources as dictionaries
        wellness_resources = {
            'Yoga': [
                'Morning Yoga Routine - <a href="https://www.example.com/yoga/morning-routine" target="_blank">Link</a>',
                'Stress-relief Yoga - <a href="https://www.example.com/yoga/stress-relief" target="_blank">Link</a>',
                'Meditation - <a href="https://www.example.com/yoga/meditation" target="_blank">Link</a>'
            ],
            'Diet Plan': [
                'Healthy Breakfast Ideas - <a href="https://www.example.com/diet/breakfast-ideas" target="_blank">Link</a>',
                'Brain-boosting Snacks - <a href="https://www.example.com/diet/brain-snacks" target="_blank">Link</a>',
                'Balanced Diet Plan - <a href="https://www.example.com/diet/balanced-plan" target="_blank">Link</a>'
            ]
        }

        # Display wellness resources
        st.markdown("### Wellness Resources")
        for category, resources in wellness_resources.items():
            st.markdown(f"#### {category}")
            for resource in resources:
                st.markdown(resource, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
