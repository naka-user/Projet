import os
import pandas as pd
import numpy as np
import streamlit as st
import io


import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Project scope",
    page_icon="📄",
)

st.sidebar.markdown("# 📄 Project scope")

st.title("Data science web application")

st.header("Part 1 : Project scope")

st.markdown(
    """
    This project is part of a data science training course context. The objective is to predict the energy performance diagnostic class of housing from its characteristics. \n
    The csv file containing the energy performance housing data comes from [Ademe open data](https://data.ademe.fr/datasets/dpe-v2-logements-neufs). \n
    Each line of the csv file corresponds a housing. Each column feature of the csv file corresponds a housing characteristics. \n
    The csv file has been modified with Excel before using the dataset for the project because it contained housing addresses and empty columns. \n
    **Steps of the project** :
    - First, we will explore the dataset to have an overview of the data.
    - Then we will study the data to understand how housings' characteristics impact the energy performance class.
    - Then we will use machine learning classification models to predict the energy performance diagnostic class of housing dataset.
"""
)




