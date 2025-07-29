import os, html, datetime, time, joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
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

DATA_DIR_RAW = "data/raw"
DATA_DIR_PROCESSED = "data/processed"
MODELS_DIR = "models"

X = pd.read_csv(os.path.join(DATA_DIR_RAW, "X_train_update.csv"), index_col=0)
X.index.name = "index"
# X_test = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, "X_test_update.csv"), index_col=0)
y = pd.read_csv(os.path.join(DATA_DIR_RAW, "Y_train_CVw08PX.csv"), index_col=0)
y.index.name = "index"
# X = X.merge(y, on = "index")

# X_ocr = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, "ocr_images_train.csv"), index_col=0)
# X_ocr.fillna("", inplace=True)
# X = X.merge(X_ocr, on = "imageid")
# X.index.name = "index"
# X["nb_tags"] = X.description.apply(lambda x : str(x).count('<'))

# X_not_num = X.select_dtypes(exclude = ['int', 'float'])
# X_num = X.select_dtypes(include = ['int', 'float'])
# X_num.replace([np.inf, -np.inf], np.nan, inplace=True)
# X_num.dropna(inplace=True)
# # # np.any(np.isnan(X_num))
# X = X_not_num.merge(X_num, on = "index")
# X.fillna("", inplace=True)
# X.drop(["productid", "imageid"], axis=1, inplace=True)

# # X = X.sample(n=5000, random_state=42)

# y = X.prdtypecode
# X = X.drop("prdtypecode", axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = X_train.select_dtypes(include = ['int', 'float'])
# X_test = X_test.select_dtypes(include = ['int', 'float'])

st.set_page_config(layout="wide")
st.title("Projet de classification multimodale de données produits - Rakuten France")
st.sidebar.title("Sommaire")
pages=["Introduction", "Exploration des données", "Modélisation", "Démo", "Conclusion", "Test"]
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
  '''
  Les colonnes *productid* et *imageid* ne portent pas de signification, ce sont seulement des identifiants.
  '''
  st.write(X.shape)
  st.dataframe(X.head(5))
  # st.dataframe(X.describe())

  st.write("#### target (y)")
  st.write(y.shape)
  st.dataframe(y.head(5))
  # st.dataframe(y.describe())


if page == "Exploration des données" : 
  st.write("## Exploration des données + augmentation")
  # Nombre de produits par catégorie
  ################################################################################
  X['designation_len'] = X['designation'].str.len()
  X['description_len'] = X['description'].str.len()
  nb_categories = y["prdtypecode"].nunique()
  st.write(f"Il y a {nb_categories} catégories..")
  y_count = y.groupby(["prdtypecode"])["prdtypecode"].count().sort_values(ascending=False)
  # display(y_count)
  st.write("Le nombre de produits par catégorie est très variable, généralement de 800 à 5000 avec 1 catégorie contenant plus de 10000 articles.")
  fig, ax = plt.subplots(figsize = (12,6))
  ax.set_title('Nombre de produits par catégorie')
  # fig = plt.figure(figsize=(10, 4))
  ax.set_ylabel('Nb de produits')
  ax.set_xlabel('Id des catégories')
  plt.xticks(rotation=45)
  sns.countplot(data = y, x="prdtypecode", order=y["prdtypecode"].value_counts().index)
  # st.pyplot(fig)
  buf = BytesIO()
  fig.savefig(buf, format="png")
  st.image(buf)


if page == "Modélisation" : 
  st.write("## Modélisation")
  # fig = plt.figure()
  # sns.countplot(x = 'Survived', data = df)
  # st.pyplot(fig)
  # fig = plt.figure()
  # sns.countplot(x = 'Sex', data = df)
  # plt.title("Répartition du genre des passagers")
  # st.pyplot(fig)

  # fig = plt.figure()
  # sns.countplot(x = 'Pclass', data = df)
  # plt.title("Répartition des classes des passagers")
  # st.pyplot(fig)

  # fig = sns.displot(x = 'Age', data = df)
  # plt.title("Distribution de l'âge des passagers")
  # st.pyplot(fig)
  # fig = plt.figure()
  # sns.countplot(x = 'Survived', hue='Sex', data = df)
  # st.pyplot(fig)

  # fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
  # st.pyplot(fig)

  # fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
  # st.pyplot(fig)

  # fig, ax = plt.subplots()
  # sns.heatmap(df.select_dtypes('number').corr(), ax=ax, cmap='RdBu_r')
  # st.write(fig)


if page == "Démo" : 
  st.write("## Démonstration")
  # df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
  # y = df['Survived']
  # X_cat = df[['Pclass', 'Sex',  'Embarked']]
  # X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
  # for col in X_cat.columns:
  #   X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
  # for col in X_num.columns:
  #   X_num[col] = X_num[col].fillna(X_num[col].median())
  # X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
  # X = pd.concat([X_cat_scaled, X_num], axis = 1)

  # from sklearn.model_selection import train_test_split
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

  # from sklearn.preprocessing import StandardScaler
  # scaler = StandardScaler()
  # X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
  # X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

  # from sklearn.ensemble import RandomForestClassifier
  # from sklearn.svm import SVC
  # from sklearn.linear_model import LogisticRegression
  # from sklearn.metrics import confusion_matrix

  # def prediction(classifier):
  #     if classifier == 'Random Forest':
  #         clf = RandomForestClassifier()
  #     elif classifier == 'SVC':
  #         clf = SVC()
  #     elif classifier == 'Logistic Regression':
  #         clf = LogisticRegression()
  #     clf.fit(X_train, y_train)
  #     return clf

  # def scores(clf, choice):
  #     if choice == 'Accuracy':
  #         return clf.score(X_test, y_test)
  #     elif choice == 'Confusion matrix':
  #         return confusion_matrix(y_test, clf.predict(X_test))

  # choix = ['Random Forest', 'SVC', 'Logistic Regression']
  # option = st.selectbox('Choix du modèle', choix)
  # st.write('Le modèle choisi est :', option)
  # clf = prediction(option)
  # import joblib
  # joblib.dump(clf, "model.joblib")
  # import pickle
  # pickle.dump(clf, open("model.pickle", 'wb'))
  # display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
  # if display == 'Accuracy':
  #     st.write(scores(clf, display))
  # elif display == 'Confusion matrix':
  #     st.dataframe(scores(clf, display))


if page == "Conclusion" : 
  st.write("## Conclusion")

if page == "Test" : 
  st.write("# titre 1")
  st.write("## titre 2")
  st.write("### titre 3")
  st.write("#### titre 4")
  st.write("##### titre 5")
  '''
  Du texte juste mis entre guillemets. Ici *entre astérisques*.
  - des tirets
  - encore des tirets

  Du code python :
  '''
  st.code(''' import streamlit ''', language='python')

st.write("2025-07-29 1125")