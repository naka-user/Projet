import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io


import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data Exploration", page_icon="📈")

st.sidebar.markdown("# 📈 Data Exploration")

csvpath = os.path.join('data','dpe.csv')
df = pd.read_csv(csvpath, sep=',', engine='python')

st.title("Data science web application")
st.header("Part 2 : Data Exploration")

st.markdown(
    """
    - The datased used for the data exploration page was modfied to have no missing values. Columns have been modified. \n
    - The global visualization of the data and relationships between variables using sns.pairplot is not used with all features because the datased has 66 features. \n
    - Duplicated rows of the dataset are not dropped because F1 score is better. \n
"""
)

st.markdown("""**Global overview of the data**""")
st.dataframe(df.head())

st.markdown("""**Shape of the data.**""")
st.write(df.shape)

st.markdown("""**Summary of the dataframe**""")

def get_df_info(df):
    buffer = io.StringIO ()
    df.info (buf=buffer)
    lines = buffer.getvalue ().split ('\n')
    # lines to print directly
    lines_to_print = [0, 1, 2, -2, -3]
    for i in lines_to_print:
        st.write (lines [i])
    # lines to arrange in a df
    list_of_list = []
    for x in lines [5:-3]:
        list = x.split ()
        list_of_list.append (list)
    info_df = pd.DataFrame (list_of_list, columns=['index', 'Column', 'Non-null-Count', 'null', 'Dtype'])
    info_df.drop (columns=['index', 'null'], axis=1, inplace=True)
    st.dataframe(info_df)

get_df_info(df)

st.markdown("""**Descriptive statistics of the dataframe**""")
st.write(df.describe())

st.markdown("""**Missing values**""") 
st.dataframe(df.isna().sum())
    
st.markdown("""**Duplicated rows**""")
st.write(df.duplicated().sum())

fig = sns.pairplot(data=df[['Cout_total_5_usages', 'Conso_eclairage_finale', 'Cout_eclairage', 'Cout_ECS', 'Etiquette_DPE']], hue="Etiquette_DPE")
plt.title("set of features to see the distribution of the data and the correlations")
st.pyplot(fig)






