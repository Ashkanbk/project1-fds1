
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.image("header.JPG", use_column_width="always")

ra=pd.read_csv('fifa.csv')
rs=pd.read_csv('results.csv')
gs=pd.read_csv('goalscorers.csv')





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

ran = ran.drop(['FIFA_Rank', 'FIFA_Points','confederation'], axis=1)



st.title('Datasets and References')



st.write("[1] **Worldwide Football Results (1872-2022)**")
st.write("   - More than 100 years of updated dataset with over 42,000 FIFA results.")
st.write("   - Source: [Link to the Dataset Source](https://www.kaggle.com/datasets/zeesolver/fifa-results)")
show_data1 = st.checkbox("Show Datasets 1")
st.text("")  # This creates a line break
st.text("")  # This creates a line break
st.text("")  # This creates a line break

# Display the DataFrame if the checkbox is checked
if show_data1:
    st.dataframe(rs)
    st.dataframe(gs)





st.write("[2] **FIFA World Ranking 1992-2023**")
st.write("   - FIFA Ranking for men's national teams from December 1992 to July 2023.")
st.write("   - Source: [Link to the Dataset Source](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)")
show_data2 = st.checkbox("Show Datasets 2")
if show_data2:
    st.dataframe(ra)
st.text("")  # This creates a line break
st.text("")  # This creates a line break
st.text("")  # This creates a line break



st.write("[3] **Data for 2021 Countries - United Nations**")
st.write("   - Data for United Nations member countries for the year 2021.")
st.write("   - Source: [Link to the Dataset Source](https://data.un.org/)")
show_data3 = st.checkbox("Show Datasets 3")
if show_data3:
    st.dataframe(ran)


