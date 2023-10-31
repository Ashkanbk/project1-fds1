import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("MT.webp", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')
gs=pd.read_csv('goalscorers.csv')
conlist=sorted(list(rs['home_team'].unique()))




# Sample DataFrame with date column
df=pd.DataFrame(pd.to_datetime(rs['date'], format='%Y-%m-%d'))

# Extract year and month
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month


# Create a Streamlit app
st.title("Number of Matches")

# Widgets for year range selection
year_start, year_end = st.slider("Select Year Range", min_value=1920, max_value=2022, value=(2015, 2022))
plot_type = st.radio("Select Plot Type", ("Yearly", "Monthly", "Average Monthly"))

# Filter the data based on the selected year range
filtered_df = df[(df['Year'] >= year_start) & (df['Year'] <= year_end)]

# Create line plots
plt.figure(figsize=(10, 6))

if plot_type == "Yearly":
    yearly_data = filtered_df['Year'].value_counts().sort_index()
    sns.lineplot(x=yearly_data.index, y=yearly_data.values,marker='o')
    plt.xlabel("Year")
    plt.ylabel("Number of Matches")
    plt.title("Yearly Number of Matches")    
    if year_start <= 1946 and year_end >= 1938:
        plt.axvspan(max(year_start, 1938), min(year_end, 1946), color='red', alpha=0.1)

    if year_start <= 2021 and year_end >= 2019:
        plt.axvspan(max(year_start, 2021), min(year_end, 2019), color='red', alpha=0.1)


elif plot_type == "Monthly":
    monthly_data = filtered_df.groupby(['Year', 'Month']).size().unstack().T
    sns.lineplot(data=monthly_data, dashes=False)
    plt.xlabel("Month")
    plt.ylabel("Number of Matches")
    plt.title("Monthly Number of Matches")
    if len(monthly_data.columns) > 10:
        plt.gca().get_legend().remove()
        
elif plot_type == "Average Monthly":
    monthly_data = filtered_df.groupby(['Year', 'Month']).size().unstack().T
    average_data = monthly_data.mean(axis=1)
    sns.lineplot(x=average_data.index, y=average_data.values, color='green',marker='o')
    plt.xlabel("Month")
    plt.ylabel("Average Matches")
    plt.title("Average Monthly Number of Matches")

# Display the plot
st.pyplot(plt)

st.write("The number of national football matches played annually exhibits historical fluctuations, including a **four-year period** pattern, and a notable **increase** on longterm average. This analysis reveals the sport's adaptability and cyclic nature, even in the face of disruptive global events, such as **World War II**, and the more recent **COVID-19 pandemic.**")

st.write("Monthly fluctuations in the number of national football matches are influenced by factors like FIFA-designated match days, the end of domestic leagues, and the **seasonality** of the sport. FIFA days and the conclusion of domestic leagues lead to an increase in matches, and the **summer season** sees a higher frequency of games due to various league schedules. These factors contribute to the dynamic nature of monthly match statistics.")

