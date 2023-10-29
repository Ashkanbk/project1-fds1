
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("H5.webp", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')
gs=pd.read_csv('goalscorers.csv')
conlist=sorted(list(rs['home_team'].unique()))




st.title('Head-to-head Statistics')

st.write("Within this section, statistics for head-to-head football matches between any two countries are made available for analysis. These statistics are categorized based on the match location, encompassing games played at **home**, **away**, in **neutral** venues, and the **total** number of encounters. For each of these categories, the number of **wins**, **draws**, and **losses** is provided, as well as the corresponding **percentages**, which can represent each team's performance under distinct conditions.")

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

# Create the bar plot
ax = subset_df.plot(kind="bar", color=selected_colors, width=0.8, figsize=(10, 6),
                    edgecolor='black', linewidth=1, legend=True)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add labels on top of the "Total" bars


if display_type == "Percent":
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom')


if display_type == "Number":
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),ha='center', va='bottom')
        


plt.ylabel(display_type)
plt.title(f'Stats of {first} in head to head matches against {second}')
plt.xticks(rotation=0)
st.pyplot(plt)

# Display the DataFrame using Streamlit
st.dataframe(newrash, use_container_width=True)



st.write("Upon close examination of these statistics, a discernible **home advantage** can be observed, especially when teams are closely matched in terms of skill and ability. This phenomenon becomes apparent when scrutinizing match results, where it becomes evident that teams tend to perform more favorably in the setting of their home ground. This observation underscores the noteworthy influence of **fan support** and the **familiarity** that teams possess with their home pitch, highlighting the challenges that come with **traveling** to face opponents on unfamiliar turf. ")
