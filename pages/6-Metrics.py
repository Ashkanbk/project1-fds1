import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("FED.png", use_column_width="always")


# Sample DataFrame with date column


ran=pd.read_csv('./Data/fifa2021.csv')
GDP1=pd.read_csv('./Data/GDP1.csv')
GDP2=pd.read_csv('./Data/GDP2.csv')
HDI=pd.read_csv('./Data/HDI.csv')
pop=pd.read_csv('./Data/pop.csv')
ran=ran[['rank','country_full','total_points','confederation']]
GDP1=GDP1[['Country or Area','Value']]
GDP2=GDP2[['Country or Area','Value']]
HDI=HDI[['Country','Value']]
pop=pop[['Country (or dependency)','Population (2020)']]
ran = ran.rename(columns={'country_full': 'Country','total_points' : 'FIFA_Points','rank' : 'FIFA_Rank'})
GDP1 = GDP1.rename(columns={'Country or Area': 'Country','Value' : 'GDP_per_capita'})
GDP2 = GDP2.rename(columns={'Country or Area': 'Country','Value' : 'GDP_gross'})
HDI = HDI.rename(columns={'Value' : 'HDI'})
pop = pop.rename(columns={'Country (or dependency)': 'Country','Population (2020)' : 'Population'})
ran = ran.merge(GDP1, on='Country')
ran = ran.merge(GDP2, on='Country')
ran = ran.merge(HDI, on='Country')
ran = ran.merge(pop, on='Country')



st.title("FIFA Confederations")
st.write("There are six confederations recognized by FIFA which oversee the game in the different continents and regions of the world.")
st.write("* **AFC**: Asian Football Confederation (47 members)")
st.write("* **CAF**: Confederation of African Football (56 members)")
st.write("* **CONCACAF**: Confederation of North, Central American and Caribbean Association Football (41 members)")
st.write("* **CONMEBOL**: Confederación Sudamericana de Fútbol (10 members)")
st.write("* **OFC**: Oceania Football Confederation (13 members)")
st.write("* **UEFA**: Union of European Football Associations (55 members)")


# Load your DataFrame
# Replace 'your_data.csv' with the actual path or URL to your data file
data = ran

st.title("Global Metrics")

st.write("This part provides an interactive tool where you can manipulate parameters for countries to investigate correlations and distributions through joint plots. Data categories include GDP, HDI, population, FIFA points, and more. Furthermore, you have the option to categorize data by confederation.")


# Select columns for x and y axes
x_column = st.selectbox("Select X-Axis Column", data.columns.drop(['Country', 'confederation']),index=4)
y_column = st.selectbox("Select Y-Axis Column", data.columns.drop(['Country', 'confederation']),index=1)

# Add logarithmic scale options

st.write("For parameters such as GDP and population, it's advisable to use a log scale when analyzing the data.")
log_x = st.checkbox("Logarithmic X-Axis", value=False)
log_y = st.checkbox("Logarithmic Y-Axis", value=False)

selected_confederations = st.multiselect("Select Confederations", data['confederation'].unique(),default=['UEFA','AFC','CAF'])
filtered_data = data[data['confederation'].isin(selected_confederations)]


# Create the scatterplot
g = sns.jointplot(data=filtered_data, x=x_column, y=y_column, kind="scatter", height=6)
f = sns.jointplot(data=filtered_data, x=x_column,y=y_column, hue="confederation")


# Set logarithmic scales if selected
if log_x:
    g.ax_joint.set_xscale('log')
    f.ax_joint.set_xscale('log')

if log_y:
    g.ax_joint.set_yscale('log')
    f.ax_joint.set_yscale('log')

st.pyplot(plt)

# Display the DataFrame for reference

# Call the update function with initial values


st.title("Definition of parameters")
st.write("* **FIFA Rank**: A ranking system that evaluates national soccer teams' performance.")
st.write("* **FIFA Points**: Points awarded to soccer teams based on their match outcomes.")
st.write("* **GDP gross**: The total value of goods and services produced in a country's economy.")
st.write("* **GDP per capita**: The economic output per person in a country.")
st.write("* **HDI**: The Human Development Index, measuring a country's overall well-being.")
st.write("* **Population**: The total number of inhabitants in a specific region or country.")
