
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("G.webp", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')
gs=pd.read_csv('goalscorers.csv')
conlist=sorted(list(rs['home_team'].unique()))















st.title('Scored Goals Distribution in a Football Game')


st.write("This section provides an insight into the distribution of scored goals in more than 20,000 football matches, minute by minute. As we explore this data through scatter and histogram plots, we discover interesting patterns that shed light on the dynamics of scoring throughout a game.")



# Sample DataFrame with scored goals and minutes
gs=gs[~(gs['minute'] > 90)]
df = gs.groupby('minute').size().reset_index(name='Goals')


# Set Seaborn style
sns.set(style="whitegrid")

# Title for the app

# Widgets in the main column
plot_type = st.radio("Select plot type", ("Scatter Plot", "Histogram"))

    




exclude_half = st.radio("Exclude minutes [1, 45, 46, 90]", ("No", "Yes"))



if exclude_half == "Yes":
    df = df[~(df['minute'] == 45) & ~(df['minute'] == 90) & ~(df['minute'] == 46) & ~(df['minute'] == 1)]
    gs = gs[~(gs['minute'] == 45) & ~(gs['minute'] == 90) & ~(gs['minute'] == 46) & ~(gs['minute'] == 1)]


# Create the plot
plt.figure(figsize=(8, 6))

highlighted_minutes = [1,45,46,90]

bin_num = st.slider("Select Number of bins", 1, 10, 6)


if plot_type == "Scatter Plot":
    sns.scatterplot(x='minute', y='Goals', data=df, color='C2')
    plt.xlabel("Minute of Game")
    plt.ylabel("Number of Scored Goals")

    for minute in highlighted_minutes:
        subset = df[df['minute'] == minute]
        sns.scatterplot(x=subset['minute'], y=subset['Goals'], color='r')

else:
    sns.histplot(data=gs ,x='minute', bins=bin_num, color='skyblue',stat="percent")
    plt.xlabel("Minute of Game")
    plt.ylabel("Percent of Scored Goals (%)")


x_ticks = [0, 15, 30, 45, 60, 75, 90]
plt.xticks(x_ticks)


# Display the plot
st.pyplot(plt)

st.write("* The scatter plots allow us to observe the number of goals scored in each minute of a football match. However, there are some **outlier** points to consider. (:question:) By excluding these outlier minutes, we can observe a clearer trend: the number of scored goals tends to **increase** with match time. As we progress toward the **end of the game**, the probability of scoring a goal also **rises**.")

st.write("* The histogram plots further illuminate these trends. They divide match time into equal **segments** and allow us to compare the percentage of scored goals in each part. These plots reveal how the **distribution** of goals changes throughout the game, highlighting when goals are most likely to be scored.")


st.markdown(''' :question:
    :blue[These outliers are due to a unique feature of the dataset: for extra minutes in each half, the last minute is recorded. For instance, a match might reach 45'+1', which is effectively 45'. This accounting method results in 45' and 90' having significantly higher goal numbers than the average. Additionally, during the first minute of each half (1' , 46'), the number of scored goals tends to be relatively lower than the average. This phenomenon can be attributed to the initial moments of a match when teams are often more cautious and focused on setting up their gameplay.] 
    ''')