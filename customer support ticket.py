import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Set Streamlit page layout
st.set_page_config(page_title="Customer Ticket Dashboard", layout="wide")
st.title("📊 Customer Support Ticket Dashboard")

# Load data from your local path
file_path = r"C:/Users/kdala/OneDrive/Documents/customer_support_tickets.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

# Drop rows without ticket description
df.dropna(subset=['Ticket Description'], inplace=True)

# Convert satisfaction to numeric
df['Customer Satisfaction Rating'] = pd.to_numeric(df['Customer Satisfaction Rating'], errors='coerce')

# Convert time columns
df['Time to Resolution'] = pd.to_timedelta(df['Time to Resolution'], errors='coerce')
df['Resolution Hours'] = df['Time to Resolution'].dt.total_seconds() / 3600

# Sentiment Analysis
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

df['Sentiment Score'] = df['Ticket Description'].apply(get_sentiment)
df['Sentiment Label'] = df['Sentiment Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Sidebar filters
st.sidebar.header("Filter Data")
sentiments = st.sidebar.multiselect("Select Sentiments", options=df['Sentiment Label'].unique(), default=df['Sentiment Label'].unique())
channels = st.sidebar.multiselect("Select Channels", options=df['Ticket Channel'].dropna().unique(), default=df['Ticket Channel'].dropna().unique())
genders = st.sidebar.multiselect("Select Genders", options=df['Customer Gender'].dropna().unique(), default=df['Customer Gender'].dropna().unique())

filtered_df = df[
    (df['Sentiment Label'].isin(sentiments)) &
    (df['Ticket Channel'].isin(channels)) &
    (df['Customer Gender'].isin(genders))
]

st.markdown(f"**Total Tickets after filtering:** {filtered_df.shape[0]}")

# First row: Sentiment breakdown + Priority by Sentiment
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Distribution")
    st.bar_chart(filtered_df['Sentiment Label'].value_counts())

with col2:
    st.subheader("Ticket Priority by Sentiment")
    priority_sentiment = filtered_df.groupby(['Ticket Priority', 'Sentiment Label']).size().unstack(fill_value=0)
    st.bar_chart(priority_sentiment)
st.subheader("Tickets by Channel")
channel_counts = filtered_df['Ticket Channel'].value_counts()
st.bar_chart(channel_counts)
# Second row: Satisfaction by Sentiment + Resolution Time by Sentiment
col3, col4 = st.columns(2)
st.subheader("Download Filtered Data")
st.download_button(
    label="📥 Download CSV",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="filtered_tickets.csv",
    mime="text/csv"
)


