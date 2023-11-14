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
import io


import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Machine Learning Models", page_icon="💻")
st.sidebar.markdown("# 💻 Machine Learning Models")

st.title("Data science web application")
st.header("Part 4 : Machine Learning Models")

st.markdown(
    """
    - Trained models are loaded and used with a new dataset to predict the energy performance diagnostic category of housing. \n
    - F1 score is used to measure models accuracy. \n
"""
)

csvpath = os.path.join('data','dataset.csv')
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
pca = PCA(n_components=0.994)
pca.fit(X)
X_pca = pca.transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0, stratify=y)

lrmodelpath = os.path.join('trained_models','lr_model.joblib')
rfmodelpath = os.path.join('trained_models','rf_model.joblib')
gbcmodelpath = os.path.join('trained_models','gbc_model.joblib')
svmmodelpath = os.path.join('trained_models','svm_model.joblib')

lr = joblib.load(lrmodelpath)
rf = joblib.load(rfmodelpath)
gbc = joblib.load(gbcmodelpath)
svm = joblib.load(svmmodelpath)

y_pred_lr=lr.predict(X_train)
y_pred_rf=rf.predict(X_train)
y_pred_svm=svm.predict(X_train)
y_pred_gbc=gbc.predict(X_train)

y_train_gb = np.zeros(len(y_train))
y_train_gb[y_train=='A']=0
y_train_gb[y_train=='B']=1
y_train_gb[y_train=='C']=2
y_train_gb[y_train=='D']=3
y_train_gb[y_train=='E']=4
y_train_gb[y_train=='F']=5
y_train_gb[y_train=='G']=6

array_dict = {0.: 'A', 1.: 'B', 2.: 'C', 3.: 'D', 4.: 'E', 5.: 'F', 6.:'G'}
y_train_gb = [array_dict[i] for i in y_train_gb]
y_pred_gbc = [array_dict[i] for i in y_pred_gbc]

choose_model = st.selectbox("Choose the model", options=['Logisitic Regression', 'Random Forest', 'XGBoost', 'SVM'], index=None, placeholder="Choose model...")

def use_model(choose_model, y_train) :

    if choose_model == 'Logisitic Regression' :
        y_pred = y_pred_lr
    
    if choose_model == 'Random Forest' :
        y_pred = y_pred_rf
    
    if choose_model == 'XGBoost' :
        y_pred = y_pred_gbc
    
    if choose_model == 'SVM' :
        y_pred = y_pred_svm
    
    return y_pred

if choose_model != None:
    st.dataframe(pd.DataFrame(classification_report(y_train, use_model(choose_model, y_train), output_dict=True)).transpose())
    st.markdown(
        """
        **Results** :
        - F1 score is not good for each class with all trained machine learning models. F1 score is less than 0.50. \n
        **Optimization** :
        - Oversampling the dataset with SMOTE to balance the class distribution. \n
        - Use neural network model.
    """
    )


