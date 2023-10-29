import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("FFF.jpg", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')
gs=pd.read_csv('goalscorers.csv')
conlist=sorted(list(rs['home_team'].unique()))









st.title("FIFA ranking ")


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
st.header("Scatter Plot")
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




st.header("Distribution of opponent rankings")
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




st.header("Distribution of opponent rankings")
fig, ax = plt.subplots(figsize=(8, 8))
sns.boxplot(data=new_data,x="Result", y="Rankdiff", hue="neutral",order=['Win','Draw','Loss'],hue_order=['Home','Neutral','Away'],palette=cupa3)
plt.legend(loc="lower left", ncol=len(new_data.columns))
st.pyplot(fig)






















