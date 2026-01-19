import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load and Preprocess Data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    df['season'] = df['season'].map(season_mapping)
    return df

df = load_data()

# Sidebar Widgets
st.sidebar.header("Filter Data")
year_filter = st.sidebar.multiselect("Select Year", options=df['year'].unique(), default=df['year'].unique())
season_filter = st.sidebar.multiselect("Select Season", options=df['season'].unique(), default=df['season'].unique())
workingday_filter = st.sidebar.radio("Working Day?", options=["Both", "Working Day", "Weekend/Holiday"])

# Filter Logic
filtered_df = df[df['year'].isin(year_filter) & df['season'].isin(season_filter)]
if workingday_filter == "Working Day":
    filtered_df = filtered_df[filtered_df['workingday'] == 1]
elif workingday_filter == "Weekend/Holiday":
    filtered_df = filtered_df[filtered_df['workingday'] == 0]

# Dashboard Title
st.title("🚲 Washington D.C. Bike Rental Dashboard")
st.markdown("This dashboard explores factors affecting bike rentals between 2011 and 2012.")

# KPI Row
col1, col2, col3 = st.columns(3)
col1.metric("Total Rentals", f"{filtered_df['count'].sum():,}")
col2.metric("Avg Hourly Rentals", round(filtered_df['count'].mean(), 2))
col3.metric("Max Rentals in an Hour", filtered_df['count'].max())

# Plot 1: Hourly Trends
st.subheader("Hourly Rental Trends by Day of Week")
fig1 = px.line(filtered_df.groupby(['hour', 'day_of_week'])['count'].mean().reset_index(), 
                x='hour', y='count', color='day_of_week', markers=True)
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Temperature vs Rentals
st.subheader("Impact of Temperature on Rentals")
fig2 = px.scatter(filtered_df, x='temp', y='count', color='weather', opacity=0.5, 
                 title="Temperature vs Total Count (Colored by Weather Category)")
st.plotly_chart(fig2, use_container_width=True)

# Plot 3 & 4: Seasonal and User Type Analysis
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Mean Rentals by Season")
    fig3 = px.bar(filtered_df.groupby('season')['count'].mean().reset_index(), x='season', y='count', color='season')
    st.plotly_chart(fig3, use_container_width=True)

with col_right:
    st.subheader("Casual vs Registered Users")
    user_counts = filtered_df[['casual', 'registered']].sum().reset_index()
    user_counts.columns = ['User Type', 'Total']
    fig4 = px.pie(user_counts, values='Total', names='User Type', hole=0.4)
    st.plotly_chart(fig4, use_container_width=True)

# Plot 5: Monthly Heatmap
st.subheader("Correlation Heatmap of Features")
plt.figure(figsize=(10, 8))
corr = filtered_df.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
st.pyplot(plt)