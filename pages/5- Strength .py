import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

st.image("FFF.jpg", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')





st.title("FIFA ranking ")

st.write("The **FIFA ranking** system is a pivotal index in international football, assessing team strength through a point-based algorithm. Teams earn or lose points based on match results and opponent strength. These rankings, announced monthly, offer a numerical representation of a team's global standing, with **higher** rankings indicating **stronger** teams. They influence key factors like tournament seeding, shaping the competitive landscape of international football.")

data = {
    'Rank': ['1', '2', '3', '4', '5'],
    'Team': ['Argentina', 'France', 'Brazil', 'England', 'Belguim'],
    'Points': [1861.29, 1853.11, 1812.20, 1807.88, 1793.71],
    'Pervious Points': [1851.41, 1840.76, 1837.61, 1794.34, 1792.64],
    '+/-': [9.88, 12.35 , -25.41 , 13.54 , 1.07],
}

rrrr = pd.DataFrame(data)
rrrr = rrrr.reset_index(drop=True)

# Display the table in Streamlit
st.dataframe(rrrr)



st.title("Scatter plot of rankings ")
st.write("In this specific scatter plot, away teams are plotted against home teams, and the teams with lower-ranking numbers are considered stronger. The plot features a y=x line that divides the domain into **two distinct regions**. Points in the upper part of the plot indicate that a team playing at home is competing against a weaker team, and conversely in the lower part. When data points cluster closer to the y=x dashed line, it signifies that the teams are more evenly matched. To further dissect the results, the data is categorized by the outcome of the **host team**, whether they lost, won, or drew. Given the extensive number of data points, a **random sample** is plotted for better visualization, and the plot's size is adjustable")


def determine_result(row):
    if row['home_score'] > row['away_score']:
        return 'Win'
    elif row['home_score'] < row['away_score']:
        return 'Loss'
    else:
        return 'Draw'
# Apply the custom function to create the 'Result' column
rs['Result'] = rs.apply(determine_result, axis=1)


radat = ra['rank_date'].unique()
rada = pd.DataFrame({'rank_date': radat})
rs['date'] = pd.to_datetime(rs['date'])
rada['rank_date'] = pd.to_datetime(rada['rank_date'])
ra['rank_date'] = pd.to_datetime(ra['rank_date'])

rada.sort_values(by='rank_date', inplace=True)
rs = pd.merge_asof(rs, rada, left_on='date', right_on='rank_date', direction='backward')
dictt = ra.set_index(['country_full', 'rank_date'])['rank'].to_dict()

rs = rs.dropna(subset=['rank_date'])


def get_rank(row, home_or_away):
    country = row[home_or_away + '_team']
    date = row['rank_date']
    
    return dictt.get((country, pd.to_datetime(date)),'NaN')

# Add columns for rank of the home and away teams
rs['HomeRank'] = rs.apply(get_rank, args=('home',), axis=1)
rs['AwayRank'] = rs.apply(get_rank, args=('away',), axis=1)

rs['HomeRank'] = pd.to_numeric(rs['HomeRank'], errors='coerce')
rs['AwayRank'] = pd.to_numeric(rs['AwayRank'], errors='coerce')

# Handle missing values (e.g., dropping rows with NaN values)
rs.dropna(subset=['HomeRank', 'AwayRank'], inplace=True)

rs['RankDiff']=rs['AwayRank']-rs['HomeRank']

data=rs

# Main content with filters
result = st.multiselect("Result", data['Result'].unique())
neutral = st.checkbox("Neutral == True ?", value=True)
filtered_data=data
if result:
    filtered_data = data[data['Result'].isin(result)]

filtered_data = filtered_data[filtered_data['neutral'] == neutral]



sample_size = st.slider("Sample Size", min_value=1, max_value=len(filtered_data), value=1000)
filtered_data = filtered_data.sample(sample_size)


# Create and display the scatter plot
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(8, 8))
x = [1, 200]
y = [1, 200]

cupa = {'Win': 'seagreen', 'Loss': 'tomato', 'Draw': 'dimgrey'}

sns.scatterplot(data=filtered_data, x="HomeRank", y="AwayRank", palette=cupa, hue="Result", ax=ax)
plt.plot(x, y, 'k--', linewidth=4)


ax.set_xlabel("Home Rank")
ax.set_ylabel("Away Rank")
ax.set_title("Scatter Plot")
ax.legend(title="Result")
ax.set_aspect('equal')

st.pyplot(fig)
st.write("It shows that for wins and losses, the plot exhibits a distinct inclination, with wins **tending to cluster** in the upper half, indicating a **higher likelihood** of winning against weaker opponents, and losses leaning towards the lower half, suggesting vulnerability against stronger teams. However, in the case of **draws**, the distribution is **symmetric**, signifying a more even outcome regardless of team strength. Notably, when considering only neutral matches, data points are **closely aligned** to the y=x line, indicating balanced contests. In contrast, for non-neutral matches, the points are more widely **dispersed**, emphasizing the significant impact of match location on team performance.")

st.title("Ranking difference ")
st.write("In this section, our focus was on introducing a means of measuring and comparing the strength of two opposing teams by calculating the **ranking difference**. I aimed to determine which team held the **ranking advantage** and how this advantage influenced match outcomes. The results, depicted in the figure below, illustrate the density distribution of ranking differences for various match contexts, including home games, away games, and neutral ground.")

gss=rs
home_data = gss[['home_team', 'RankDiff', 'Result','neutral','date']]
home_data.columns = ['team', 'Rankdiff', 'Result','neutral','date']
replace_dict = {True: 'Neutral', False: 'Home'}
home_data['neutral'] = home_data.apply(lambda row: replace_dict.get(row['neutral'], row['neutral']), axis=1)

# For the away team
away_data = gss[['away_team', 'RankDiff', 'Result','neutral','date']]
away_data.columns = ['team', 'Rankdiff', 'Result','neutral','date']
replace_dict = {True: 'Neutral', False: 'Away'}
away_data['neutral'] = away_data.apply(lambda row: replace_dict.get(row['neutral'], row['neutral']), axis=1)


away_data['Rankdiff']=-away_data['Rankdiff']

replace_dict = {'Win': 'Loss', 'Loss': 'Win'}
away_data['Result'] = away_data.apply(lambda row: replace_dict.get(row['Result'], row['Result']), axis=1)

# Concatenate the data for home and away teams
new_data = pd.concat([home_data, away_data])

# Reset the index for the new DataFrame
new_data.reset_index(drop=True, inplace=True)

cupa2 = {'Win': 'green', 'Loss': 'red', 'Draw': 'k'}
cupa3 = {'Home': 'c', 'Neutral': 'grey', 'Away': 'tomato'}



neutral_filter  = st.radio("Select Neutral Filter:", ["Home", "Away", "Neutral"])

filtered_data=new_data[new_data['neutral'] == neutral_filter ]

fig, ax = plt.subplots(figsize=(8, 8))
sns.kdeplot(data=filtered_data, x="Rankdiff",hue="Result",palette=cupa2,ax=ax,fill=True)

st.pyplot(fig)

st.write("Notably, the figure reveals a trend: when a team has a **ranking advantage** and plays on their home, the **probability** of winning the game significantly **increases**. This emphasizes the significance of both team strength and the home advantage in determining match outcomes. Additionally, the **symmetric distribution** of ranking differences for draws stands out as an interesting observation, suggesting that the ranking difference **alone might not be as influential** in games ending in a draw, and other factors come into play in such cases.")
st.write("Box plots and violin plots are alternative visual representations thae above plot effectively illustrate data distributions, with box plots highlighting summary statistics")
plot_type = st.radio("Select Plot Type", ["Violin Plot", "Box Plot"])
fig, ax = plt.subplots(figsize=(8, 8))

if plot_type == "Box Plot":
    sns.boxplot(data=new_data,x="Result", y="Rankdiff", hue="neutral",order=['Win','Draw','Loss'],hue_order=['Home','Neutral','Away'],palette=cupa3)
    plt.legend(loc="lower left", ncol=len(new_data.columns))
    st.pyplot(fig)


if plot_type == "Violin Plot":
    sns.violinplot(data=new_data,x="Result", y="Rankdiff", hue="neutral",order=['Win','Draw','Loss'],hue_order=['Home','Neutral','Away'],palette=cupa3)
    plt.legend(loc="lower left", ncol=len(new_data.columns))
    st.pyplot(fig)



















