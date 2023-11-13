import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io


import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Data Visualization", page_icon="📊")
st.sidebar.markdown("# 📊 Data Visualization")

csvpath = os.path.join('data','dpe.csv')
df = pd.read_csv(csvpath, sep=',', engine='python')

st.title("Data science web application")
st.header("Part 3 : Data Visualization")

test = df['Etiquette_DPE'].value_counts() / len(df) * 100
values = np.array(df['Etiquette_DPE'].value_counts() / len(df) * 100)
labels = np.array(test.index)

fig = plt.figure(figsize=(8,8))
plt.pie(values, labels=labels, autopct='%1.1f%%')
plt.title("Distribution of energy performance diagnostic in %")
st.pyplot(fig)

fig2 = plt.figure(figsize=(8,8))
sns.countplot(data = df, x = 'Etiquette_DPE', order = df['Etiquette_DPE'].value_counts().index)
plt.title("Distribution of energy performance diagnostic")
st.pyplot(fig2)

fig3 = plt.figure(figsize=(8,8))
sns.countplot(data = df, x = 'Type_batiment', order = df['Type_batiment'].value_counts().index, hue = 'Etiquette_DPE')
plt.title("Distribution of energy performance diagnostic by building type")
st.pyplot(fig3)

fig4 = plt.figure(figsize=(8,8))
sns.countplot(data = df, x = 'Qualite_isolation_enveloppe', order = df['Qualite_isolation_enveloppe'].value_counts().index, hue = 'Etiquette_DPE')
plt.xticks(rotation=90)
plt.title("Distribution of energy performance diagnostic by Qualite_isolation_enveloppe")
st.pyplot(fig4)

fig5 = plt.figure(figsize=(8,8))
sns.countplot(data = df, x = 'Type_energie_num1', order = df['Type_energie_num1'].value_counts().index, hue = 'Etiquette_DPE')
plt.xticks(rotation=90)
plt.title("Distribution of energy performance diagnostic by Type_energie_num1")
st.pyplot(fig5)

fig6 = plt.figure(figsize=(8,8))
sns.scatterplot(x="Cout_total_5_usages", y="Cout_ECS_energie_num1", data=df, hue="Etiquette_DPE")
plt.title("Distribution of energy performance diagnostic by Cout_total_5_usages and Cout_ECS_energie_num1")
st.pyplot(fig6)

st.markdown(
    """
    - The most represented energy performance diagnostic categories are A, B and C. \n
    - Most housing with energy performance diagnostic A are houses. \n
    - Most housing with energy performance diagnostic A use electricity. \n
    - Most housing with energy performance diagnostic C use natural gas. \n
"""
)
