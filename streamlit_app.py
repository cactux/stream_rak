import os, html, datetime, time, joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(layout="wide")
st.title("Projet de classification multimodale de données produits - Rakuten France")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration", "DataVizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

if page == "Introduction" : 
  st.write("## Introduction")
  '''
  Les données viennent de la plateforme Challenge Data, organisée par l’ENS. Elles sont obtenues depuis https://challengedata.ens.fr/challenges/35 et ont été fournies par la société Rakuten. Cette société édite un site web de commerce en ligne où les utilisateurs peuvent vendre eux-mêmes leurs objets neufs, d’occasion ou reconditionnés.

À partir de données texte et image sur plus de 84 000 produits, l’objectif est de déterminer le bon type du produit. Les données sont à priori très spartiates :
- 1 titre plutôt court, disponible pour 100 % des entrées.
- 1 description pouvant être plus longue, présente pour environ 2 tiers des entrées.
- 1 fichier image JPEG.
- Le type de produit, parmi une liste de 27 possibles.

La catégorie des 84 000 produits est connue : nous avons de **données labellisées** : il s’agit d’un sujet de machine learning supervisé. Il faut déterminer le type d’un produit : il s’agit d’un problème de **classification**.


  TODO : insérer ici des exemples d'images et de texte ?
  '''

  st.write("#### Features (X)")

  st.write("#### target (y)")


if page == "Exploration" : 
  st.write("## Exploration")


if page == "DataVizualization" : 
  st.write("## DataVizualization")


if page == "Modélisation" : 
  st.write("## Modélisation")

st.write("2025-07-28 1624")
