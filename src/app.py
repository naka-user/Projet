import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import pickle
import streamlit as st


import warnings

warnings.filterwarnings("ignore")


csvpath = os.path.join('..','data','dpe.csv')

df = pd.read_csv(csvpath, sep=',', engine='python')

st.dataframe(df.head())

st.sidebar.title("Sommaire")

pages = ["Project context", "Exploratory data analysis", "Modelization"]

page = st.sidebar.radio("Go to the page :", pages)

if page == pages[0] : 
    
    st.write("### Projet context")
    
    st.write("This project is part of a end of data science training project context. The objective is to predict the energy performance class of a housing from its characteristics.")
    
    st.write("The csv file containing the energy performance housing data comes from the Ademe open data website.")
             
    st.write("Each line of the csv file corresponds a housing. Each column feature of the csv file corresponds a housing characteristics.")

    st.write("The csv file has been modified with Excel before using the dataset for the project because it contained housing addresses and empty columns")
    
    st.write("First, we will explore the dataset to have an overview of the data. Then we will study the data to understand how housings' characteristics impact the energy performance class. Then we will use machine learning classification models to predict the energy performance class of a housing.")
    
    st.image("immobilier.jpg")
    
elif page == pages[1]:
    st.write("### Exploratory data analysis")
    
    st.dataframe(df.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(df.shape)
    
    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())