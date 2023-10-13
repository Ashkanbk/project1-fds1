import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("header.JPG", use_column_width="always")

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
st.title("Football Match Analysis")

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














st.title('History of matches')

col1, col2 = st.columns(2)

with col1:
    first = st.selectbox('First Team:',conlist, index=conlist.index('England'))


with col2:
    second = st.selectbox('Second Team:',conlist, index=conlist.index('Italy'))


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
combined_df = combined_df.fillna(0)
combined_df['Total'] = combined_df['Home'] + combined_df['Away'] + combined_df['Neutral']
desired_order = ['Win', 'Draw', 'Lose']
combined_df = combined_df.reindex(desired_order)


# Radio button for selecting number or percent
display_type = st.radio("Display Type", ["Number", "Percent"], horizontal=True)

# Checkbox for selecting columns
st.write("Select Columns")

col1, col2, col3 ,col4  = st.columns(4)
with col1:
    show_home = st.checkbox("Home",value=True)
with col2:
    show_away = st.checkbox("Away",value=True)
with col3:
    show_neutral = st.checkbox("Neutral",value=True)
with col4:
    show_total = st.checkbox("Total",value=True)



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

ax = subset_df.plot(kind="bar", color=selected_colors, width=0.8, figsize=(10, 6)
, edgecolor='black', linewidth=1, legend=True)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylabel(display_type)
plt.title(f'Stats of {first} in head to head matches with {second}')
plt.xticks(rotation=0)
st.pyplot(plt)

# Display the DataFrame using Streamlit
st.dataframe(newrash, use_container_width=True)
























# Sample DataFrame with scored goals and minutes
gs=gs[~(gs['minute'] > 90)]
df = gs.groupby('minute').size().reset_index(name='Goals')


# Set Seaborn style
sns.set(style="whitegrid")

# Title for the app
st.title("Goals vs Minute of Game")

# Widgets in the main column
bin_size = st.slider("Select bin size", 1, 45, 5)
exclude_half = st.radio("Exclude minutes [1, 45, 46, 90]", ("No", "Yes"))
plot_type = st.radio("Select plot type", ("Scatter Plot", "Histogram"))

if exclude_half == "Yes":
    df = df[~(df['minute'] == 45) & ~(df['minute'] == 90) & ~(df['minute'] == 46) & ~(df['minute'] == 1)]

# Create the plot
plt.figure(figsize=(8, 6))

if plot_type == "Scatter Plot":
    sns.scatterplot(x='minute', y='Goals', data=df, color='coral')
    plt.xlabel("Minute of Game")
    plt.ylabel("Number of Scored Goals")
else:
    sns.histplot(data=gs ,x='minute', bins=range(0, 90, bin_size), color='skyblue',stat="percent")
    plt.xlabel("Minute of Game")
    plt.ylabel("Number of Scored Goals")

# Display the plot
st.pyplot(plt)










st.title("Football Match Analysis")


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




st.header("kde plot")
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




st.header("box plot")
fig, ax = plt.subplots(figsize=(8, 8))
sns.boxplot(data=new_data,x="Result", y="Rankdiff", hue="neutral",order=['Win','Draw','Loss'],hue_order=['Home','Neutral','Away'],palette=cupa3)
plt.legend(loc="lower left", ncol=len(new_data.columns))
st.pyplot(fig)















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




import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your DataFrame
# Replace 'your_data.csv' with the actual path or URL to your data file
data = ran

st.title("Scatterplot Visualization")

# Select columns for x and y axes
x_column = st.selectbox("Select X-Axis Column", data.columns.drop(['Country', 'confederation']))
y_column = st.selectbox("Select Y-Axis Column", data.columns.drop(['Country', 'confederation']))

# Add logarithmic scale options
log_x = st.checkbox("Logarithmic X-Axis", value=False)
log_y = st.checkbox("Logarithmic Y-Axis", value=False)

selected_confederations = st.multiselect("Select Confederations", data['confederation'].unique())
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








