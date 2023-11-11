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
import streamlit as st
from sklearn.metrics import r2_score


import warnings

warnings.filterwarnings("ignore")


csvpath = os.path.join('..','data','dpe.csv')

df = pd.read_csv(csvpath, sep=',', engine='python')

st.sidebar.title("Sommaire")

pages = ["Project context", "Data exploration", "Data analysis", "Modelization"]

page = st.sidebar.radio("Go to the page :", pages)

if page == pages[0] : 
    
    st.write("### Projet context")
    
    st.write("This project is part of a end of data science training project context. The objective is to predict the energy performance class of a housing from its characteristics.")
    
    st.write("The csv file containing the energy performance housing data comes from the Ademe open data website.")
             
    st.write("Each line of the csv file corresponds a housing. Each column feature of the csv file corresponds a housing characteristics.")

    st.write("The csv file has been modified with Excel before using the dataset for the project because it contained housing addresses and empty columns")
    
    st.write("First, we will explore the dataset to have an overview of the data. Then we will study the data to understand how housings' characteristics impact the energy performance class. Then we will use machine learning classification models to predict the energy performance class of a housing.")
    
    #st.image("immobilier.jpg")
    
elif page == pages[1]:
    st.write("### Data exploration")

    st.dataframe(df.head())

    st.write("Global overview of the data.")

    st.write("Shape of the data.")
    
    st.write(df.shape)

    st.write("Summary of the dataframe.")

    st.write(df.info())

    st.write("Descriptive statistics of the dataframe.")

    st.write(df.describe())

    if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
        
    if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())
    
    # fig = sns.pairplot(df, hue='Etiquette_DPE')
    # plt.title("set of features to see the distribution of the data and the correlations")
    # st.pyplot(fig)

elif page == pages[2]:
    st.write("### Data analysis")
    
    # fig = sns.displot(x='price', data=df, kde=True)
    # plt.title("Distribution de la variable cible price")
    # st.pyplot(fig)
    
    # fig2 = px.scatter(df, x="price", y="area", title="Evolution du prix en fonction de la surface")
    # st.plotly_chart(fig2)
    
    # fig3, ax = plt.subplots()
    # sns.heatmap(df.corr(), ax=ax)
    # plt.title("Matrice de corrélation des variables du dataframe")
    # st.write(fig3)

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

    # fig6, ax = plt.subplots()
    # sns.heatmap(df.corr(), ax=ax)
    # plt.title("Corrélation matrix")
    # st.write(fig6)

elif page == pages[3]:
    st.write("### Modelization")
    
    csvpath = os.path.join('..','data','df_preprocessed.csv')

    df_prep = pd.read_csv(csvpath, sep=',', engine='python')
    
    numerical_columns = ['Hauteur_sous_plafond',
       'Conso_5_usages_finale', 'Conso_5_usages_m2_finale',
       'Conso_chauffage_finale', 'Conso_chauffage_depensier_finale',
       'Conso_eclairage_finale', 'Conso_ECS_finale',
       'Conso_ECS_depensier_finale', 'Conso_refroidissement_finale',
       'Conso_refroidissement_depensier_finale', 'Conso_auxiliaires_finale',
       'Conso_5_usages_primaire', 'Conso_5_usages_par_m2_primaire',
       'Conso_chauffage_primaire', 'Conso_chauffage_depensier_primaire',
       'Conso_eclairage_primaire', 'Conso_ECS_primaire',
       'Conso_ECS_depensier_primaire', 'Conso_refroidissement_primaire',
       'Conso_refroidissement_depensier_primaire',
       'Conso_auxiliaires_primaire', 'Emission_GES_5_usages',
       'Emission_GES_5_usages_par_m2', 'Emission_GES_chauffage',
       'Emission_GES_chauffage_depensier', 'Emission_GES_eclairage',
       'Emission_GES_ECS', 'Emission_GES_ECS_depensier',
       'Emission_GES_refroidissement',
       'Emission_GES_refroidissement_depensier', 'Emission_GES_auxiliaires',
       'Cout_chauffage_energie_num1', 'Cout_ECS_energie_num1',
       'Emission_GES_5_usages_energie_num1',
       'Cout_total_5_usages', 'Cout_chauffage', 'Cout_chauffage_depensier',
       'Cout_eclairage', 'Cout_ECS_depensier', 'Cout_ECS',
       'Cout_refroidissement', 'Cout_refroidissement_depensier',
       'Cout_auxiliaires',
       'Ubat_W_m2_K', 'Nombre_appartement',
       'Nombre_niveau_immeuble', 'Nombre_niveau_logement',
       'Surface_habitable_immeuble', 'Surface_habitable_logement', 'Score_BAN',
       'Cout_chauffage_energie_num2', 'Cout_ECS_energie_num2', 'Emission_GES_5_usages_energie_num2']
    
    categorical_columns = ['Modele_DPE', 'Version_DPE', 'Methode_application_DPE', 
                      'Type_batiment', 'Type_energie_num1', 'Qualite_isolation_enveloppe',
                      'Qualite_isolation_menuiseries', 'Qualite_isolation_murs', 'Appartement_non_visite_0_1',
                      'Type_energie_num2', 'Qualite_isolation_plancher_bas']
    
    # Dummies
    to_dummify = df_prep[categorical_columns]
    dummies = pd.get_dummies(to_dummify)
    dummies = dummies.drop(['Appartement_non_visite_0_1_FAUX'], axis=1)

    # Scaling
    to_scale = df_prep[numerical_columns]
    to_scale = (to_scale - to_scale.mean(axis=0))/(to_scale.std(axis=0))

    X = pd.concat([to_scale, dummies], axis=1)
    y = df_prep['Etiquette_DPE']
    
    # PCA
    pca = PCA()
    pca.fit(X)
    X_pca = pca.transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0, stratify=y)
    
    lr = joblib.load("lr_model.joblib")
    rf = joblib.load("rf_model.joblib")
    gbc = joblib.load("gbc_model.joblib")
    svm = joblib.load("svm_model.joblib")
    
    y_pred_lr=lr.predict(X_test)

    y_pred_rf=rf.predict(X_test)

    y_pred_svm=svm.predict(X_test)
    
    y_pred_gbc=gbc.predict(X_test)

    y_test_gb = np.zeros(len(y_test))
    y_test_gb[y_test=='A']=0
    y_test_gb[y_test=='B']=1
    y_test_gb[y_test=='C']=2
    y_test_gb[y_test=='D']=3
    y_test_gb[y_test=='E']=4
    y_test_gb[y_test=='F']=5
    y_test_gb[y_test=='G']=6

    array_dict = {0.: 'A', 1.: 'B', 2.: 'C', 3.: 'D', 4.: 'E', 5.: 'F', 6.:'G'}
    y_test_gb = [array_dict[i] for i in y_test_gb]
    y_pred_gbc = [array_dict[i] for i in y_pred_gbc]
    
    choose_model = st.selectbox(label = "Model", options = ['Logisitic Regression', 'Random Forest', 'XGBoost', 'SVM'])
    
    def train_model(choose_model, y_test) : 
        if choose_model == 'Logisitic Regression' :
            y_pred = y_pred_lr
        elif choose_model == 'Random Forest' :
            y_pred = y_pred_rf
        elif choose_model == 'XGBoost' :
            y_pred = y_pred_gbc
        elif choose_model == 'SVM' :
            y_pred = y_pred_svm
        
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    
    st.write("F1 score", train_model(choose_model, y_test))






