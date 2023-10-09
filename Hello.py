import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("header.JPG", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')
gs=pd.read_csv('goalscorers.csv')


### st.title('1- Raw data')

_ ='''
dbs = st.radio(
    "What's your favorite movie genre",
    ["ra", "rs", "gs"],
    horizontal=True
)
if dbs == "ra":
    st.dataframe(ra)
elif dbs == "rs":
    st.dataframe(rs)
elif dbs == "gs":
    st.dataframe(gs)
st.dataframe(gs)
'''


st.title('History of matches')

col1, col2 = st.columns(2)

with col1:
    first = st.selectbox(
        'First Team:',
        ('England', 'France', 'Italy')
    )


with col2:
    second = st.selectbox(
        'Second Team:',
        ('France', 'England', 'Italy')
    )


newra=rs[((rs['home_team'] == first) & (rs['away_team'] == second))|((rs['home_team'] == second) & (rs['away_team'] == first))]

newrash=newra[['date','home_team','home_score','away_score','away_team','tournament','city']]



def calculate_result(row):
    if row['home_score'] > row['away_score']:
        return 'Win'
    elif row['home_score'] < row['away_score']:
        return 'Lose'
    else:
        return 'Draw'

newra['result'] = newra.apply(calculate_result, axis=1)

T1=newra[(newra['home_team']==first) & (newra['neutral']==False)]
T1['result'].value_counts()
T2=newra[(newra['away_team']==first) & (newra['neutral']==False)]
replacements = {'Lose': 'Win', 'Win': 'Lose'}
T2['result'] = T2['result'].replace(replacements)
T2['result'].value_counts()
T3=newra[newra['neutral']==True]
T3['result'].value_counts()
df_T1 = pd.DataFrame(T1['result'].value_counts())
df_T2 = pd.DataFrame(T2['result'].value_counts())
df_T3 = pd.DataFrame(T3['result'].value_counts())
combined_df = pd.concat([df_T1, df_T2, df_T3], axis=1)
combined_df.columns = ['Home', 'Away', 'Neutral']
combined_df['Total'] = combined_df['Home'] + combined_df['Away'] + combined_df['Away']



# Radio button for selecting number or percent
display_type = st.radio("Display Type", ["Number", "Percent"], horizontal=True)

# Checkbox for selecting columns
st.write("Select Columns")

col1, col2, col3 ,col4  = st.columns(4)
with col1:
    show_home = st.checkbox("Home")
with col2:
    show_away = st.checkbox("Away")
with col3:
    show_neutral = st.checkbox("Neutral")
with col4:
    show_total = st.checkbox("Total")



# Define custom colors for columns
column_colors = {
    'Home': 'slateblue',
    'Away': 'chocolate',
    'Neutral': 'gold',
    'Total': 'gray'
}

# Filter the columns to display
selected_columns = []
selected_colors = []
if show_home:
    selected_columns.append('Home')
    selected_colors.append(column_colors['Home'])
if show_away:
    selected_columns.append('Away')
    selected_colors.append(column_colors['Away'])
if show_neutral:
    selected_columns.append('Neutral')
    selected_colors.append(column_colors['Neutral'])
if show_total:
    selected_columns.append('Total')
    selected_colors.append(column_colors['Total'])

# Create a subset DataFrame based on the selected columns
subset_df = combined_df[selected_columns]

# Calculate percentages if needed
if display_type == "Percent":
    subset_df = subset_df.div(subset_df.sum(axis=0), axis=1) * 100

# Create a grouped bar plot with custom colors
plt.figure(figsize=(10, 6))
categories = subset_df.index
columns = subset_df.columns
bar_width = 0.2
index = range(len(categories))

for i, (col, color) in enumerate(zip(columns, selected_colors)):
    plt.bar([x + i * bar_width for x in index], subset_df[col], bar_width, label=col, color=color, edgecolor='black', linewidth=1)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.ylabel(display_type)
plt.title(f'Stats of {first} in head to head matches with {second}')
plt.xticks([x + bar_width for x in index], categories)
plt.legend()

# Display the plot in Streamlit
st.pyplot(plt)


st.dataframe(newrash,use_container_width=True)






dfEn=ra[ra['country_full'] == 'England']
dfFR=ra[ra['country_full'] == 'France']
dfIt=ra[ra['country_full'] == 'Italy']

st.title('Fifa Ranking')

show_curve1 = st.checkbox('England', value=True)
show_curve2 = st.checkbox('France', value=True)
show_curve3 = st.checkbox('Italy', value=True)

fig, ax = plt.subplots()
if show_curve1:
    plt.plot(dfEn['rank_date'], dfEn['rank'],label='England')
if show_curve2:
    plt.plot(dfFR['rank_date'], dfFR['rank'],label='France')
if show_curve3:
    plt.plot(dfIt['rank_date'], dfIt['rank'],label='Italy')

plt.xlabel('Date')
plt.ylabel('Ranking')
new_ticks = [1, 50, 100, 150,200, 250 ,290, 326]  # Specify the tick positions
new_labels = ['1992', '1998', '2002', '2007', '2011','2015', '2019', '2023']  # Specify the corresponding labels
plt.xticks(new_ticks, new_labels)
plt.title('Ranking over time')
plt.legend()
st.pyplot(fig)






import streamlit as st
import folium
from folium.plugins import MiniMap
from geopy.geocoders import Nominatim

# List of cities to plot
cities = ["New York", "Los Angeles", "Chicago", "London", "Paris", "Sydney"]

# Initialize a geocoder
geolocator = Nominatim(user_agent="city_plotter")

# Create a Folium map centered around the world
m = folium.Map(location=[0, 0], zoom_start=2)

# Plot the city locations
for city_name in cities:
    location = geolocator.geocode(city_name)
    
    if location:
        latitude = location.latitude
        longitude = location.longitude
        folium.Marker([latitude, longitude], tooltip=city_name).add_to(m)

# Add a minimap for navigation (optional)
minimap = MiniMap(toggle_display=True)
m.add_child(minimap)

# Convert the Folium map to HTML
folium_map_html = m._repr_html_()

# Display the map in the Streamlit app
st.write("City Locations on World Map")
st.components.v1.html(folium_map_html, width=800, height=600)



# Sample data
data = {
    'Group': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Category': ['Category 1', 'Category 2', 'Category 1', 'Category 2', 'Category 1', 'Category 2'],
    'Value': [10, 15, 12, 18, 8, 10]
}

df = pd.DataFrame(data)

# Streamlit app
st.title('Grouped Barplots with Checkboxes')

# Checkboxes for group selection
selected_groups = st.multiselect('Select Groups', df['Group'].unique())

# Filter the DataFrame based on selected groups
filtered_df = df[df['Group'].isin(selected_groups)]

# Create a grouped barplot
if not filtered_df.empty:
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x='Category', y='Value', hue='Group', ax=ax)
    st.pyplot(fig)
else:
    st.warning('No data to display for the selected groups.')