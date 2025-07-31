import os, html, datetime, time, joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
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

DATA_DIR_RAW = ""
DATA_DIR_PROCESSED = "data/processed"
MODELS_DIR = "models"

X = pd.read_csv(os.path.join(DATA_DIR_RAW, "X_train_update.csv"), index_col=0)
X.index.name = "index"
# X_test = pd.read_csv(os.path.join(DATA_DIR_PROCESSED, "X_test_update.csv"), index_col=0)
y = pd.read_csv(os.path.join(DATA_DIR_RAW, "Y_train_CVw08PX.csv"), index_col=0)
y.index.name = "index"
# X = X.merge(y, on = "index")
lang_list = np.genfromtxt("lang_list.csv", dtype = None).tolist()
lang_dist = pd.read_csv("lang_dist.csv", index_col = 0).rename({"0": "proportion"}, axis = 1)
lang_dist.index.name = "langue"

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
pages=["Introduction", "Exploration des données", "Modélisation - meta-data", "Modélisation - images", "Modélisation - textes", "Démonstration", "Conclusion", "Page de test"]
page=st.sidebar.radio("Aller vers", pages)


########################################################## Introduction ###########################################################
if page == "Introduction" : 
  st.write("## Introduction")
  '''
  Dans le cadre de la formation Data Scientist, suivie chez DataScientest en mode cursus continu promotion NOV24, nous avons mené un projet de Data Science.
  Les données viennent de la plateforme Challenge Data, organisée par l'ENS. Elles sont obtenues depuis https://challengedata.ens.fr/challenges/35 et ont été fournies par la société Rakuten.
  Cette société édite un site web de commerce en ligne où les utilisateurs peuvent vendre eux-mêmes leurs objets neufs, d'occasion ou reconditionnés. Ces produits sont rangés dans des catégories.

  ### Objectif

  À partir de données texte et image sur plus de 84 000 produits, l'objectif est de déterminer le bon type du produit. Les données fournies sont à priori très spartiates :
  - 1 titre plutôt court, disponible pour 100 % des entrées : *designation*.
  - 1 description pouvant être plus longue, présente pour environ 2 tiers des entrées : *description*
  - 1 fichier image JPEG.
  - Le code du type de produit, parmi une liste de 27 possibles.

  La catégorie des 84 000 produits est connue : nous avons des **données labellisées** : il s'agit d'un sujet de machine learning supervisé.
  Il faut déterminer le type d'un produit : il s'agit d'un problème de **classification**.

  ### Quelques exemples de données :

  Avant de plonger dans l'analyse exploratoire des données, voici quelques exemples :

  - Produit 70344272 : (imageid = 864552991)
    - Designation = Esprit D Hurlorage La Traque D Illidan Wow Vf Epique
    - Description = nan
  - Produit 4165063798 : (imageid = 1305321064)
    - Designation = Modèle De Voituregpm Racing Aluminum Front/Rear Adjustable Spring Damper For Ttaxxas Trx-4 Rc Car Hgf90712001gy-Générique
    - Description = GPM Racing Aluminum Front/Rear Adjustable Spring Damper For TTAXXAS TRX-4 Rc Car  Description?  Outer diameter of the spring coil of the spring shock cylinder shock shaft are all made thicker   Reduce the friction and absorb the pressure during driving so that the car can be more balanced and improves the crawling   outer diameter of spring damper from original 16mm increase to 17mm    The cylinder of the shock absorber from original 11mm to 12.8mm   Spring coil from original 1.1mm increased to 1.2mm   Shock shaft from 3mm increased to 3.2mm  We use a quality light weight high tensile strength aluminum 6061 T6 and is then anodized with scratch resistant colors   Package include: 2xSTAINLESS STEEL ROUND HEAD SCREWS 3x23mm 2xSTAINLESS STEEL ROUND HEAD SCREWS 3x27mm 2xShock Absorbers
  - Produit 2231547157 : (imageid = 1132713850)
    - Designation = Alsace - France - Plaquette Depliante Touristique
    - Description = nan
  - Produit 1418192832 : (imageid = 1148427461)
    - Designation = L'univers Illustre - Trente Deuxieme Annee N° 1790 - Paris La Vente De La Galerie Secrétan : Adjudication De L'angelus De J.-F. Millet (Dessin D'après Nature De M.Paul Destez)
    - Description = Journal hebdomadaire. Sommaire : la catastrophe d'Aubervilliers - Paris - Inauguration de la statue de la liberté éclairant le monde à Grenelle dessin d'après nature de M.Guilliod) - Paris - Exposition universelle : entre quatre et cinq promenade en fauteuil roulant (dessin d'après nature de M.Paul Destez) - Les courses de taureau en Espagne - A l'exposition par ci par là - Les manoeuvres navales dans la méditerranée  La défense de Toulon : Exercices de tir en mer par la batterie du polygone (dessin d'après nature de M.Riou) - Le cuirassé grade-côte La fusée - La catastrophe d'Aubervilliers (dessin d'après nature de M.Paul Merwaert).
  '''
  st.image('rakuten-exemple-4.jpg')


########################################################## Exploration des données ###########################################################
if page == "Exploration des données" : 
  st.write("## Exploration des données + augmentation")
  '''
  Les données fournies sont :
- 3 fichiers CSV :
  - X_test_update.csv : 13 813 lignes
  - X_train_update.csv : 84 920 lignes
  - Y_train_CVw08PX.csv : 84 917 lignes
- 1 dossier images avec 2 sous-dossiers :
  - images/image_test : 13 812 images au format JPG
  - images/image_train : 84 916 images au format JPG

Pour le projet DataScientest, nous utiliserons les fichiers CSV `X_train_update.csv` et `Y_train_CVw08PX.csv`, ainsi que les images du dossier `image_train`
(le fichier `X_test_update.csv` et le dossier `image_test` sont utilisés pour participer au challenge).

Les données sont analysées à l'aide de notebooks python et scripts bash, disponibles dans le projet github https://github.com/DataScientest-Studio/NOV24-CDS-RAKUTEN.
L'analyse des fichiers CSV est dans le fichier Rapport exploration des données - Projet Rakuten.xlsx basé sur le template fourni.

### Fichier X_train_update.csv
infos :\n
Index: 84916 entries, 0 to 84915\n
Data columns (total 4 columns):\n
` #   Column       Non-Null Count  Dtype `\n
`---  ------       --------------  ----- `\n
` 0   designation  84916 non-null  object`\n
` 1   description  84916 non-null  object`\n
` 2   productid    84916 non-null  int64 `\n
` 3   imageid      84916 non-null  int64`\n

#### Valeurs NaN
`designation        0`\n
`description    29800`\n
`productid          0`\n
`imageid            0`\n

Près d’un tiers des lignes ont leur colonne *description* vide.

### Fichier Y_train_CVw08PX.csv 
Infos :\n
Index: 84916 entries, 0 to 84915\n
Data columns (total 1 columns):\n
` #   Column       Non-Null Count  Dtype`\n
`---  ------       --------------  -----`\n
` 0   prdtypecode  84916 non-null  int64`\n

On a autant de lignes que dans le fichier X_train_update.csv, ce fichier contient la cible à prédire : le code catégorie (type de produit).

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

  # Nombre de produits par catégorie
  X['designation_len'] = X['designation'].str.len()
  X['description_len'] = X['description'].str.len()
  nb_categories = y["prdtypecode"].nunique()
  st.write(f"Il y a {nb_categories} catégories..")
  y_count = y.groupby(["prdtypecode"])["prdtypecode"].count().sort_values(ascending=False)
  # display(y_count)
  '''
  Le nombre de produits par catégorie est très variable, généralement de 800 à 5000 avec 1 catégorie contenant plus de 10000 articles.
  La catégorie ayant le plus de produits en a environ le double de chacune des 7 catégories suivantes : c’est à surveiller pour l'entraînement des modèles.
  De même, les 6 catégories ayant le moins de produits en ont environ 5 fois moins.'''
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
  
  '''
  #### `Proportion des produits par catégorie`'''
 
  st.write(y.value_counts(ascending = False, normalize = True).map("{:0.2%}".format), "\n")
  '''Ce tableau nous renvoit une distribution **non uniforme** des catégories: la catégorie **2583** embarque à elle seule
  **12.02%** des observations tandis que chacune des **26** catégories restantes se partagent le reliquat du dataset 
  dans une proportion allant de **~1%** à **~6%** 
  => il s'agit donc d'un problème de **classification déséquilibrée**'''
  

  st.write("#### feature (designation)")
  st.write("Répartition de la longueur des désignations.")
  '''Le champ 'désignation' contient tout le temps du texte, le plus court faisant 11 caractères.'''
  fig, ax = plt.subplots(figsize = (12,6))
  ax.set_title('Répartition de la longueur des designations')
  # fig = plt.figure(figsize=(10, 4))
  ax.set_ylabel("Nb d'occurences")
  ax.set_xlabel('Longueur en caractères')
  plt.xticks(ticks=range(10, 250, 10))#, labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
  plt.hist(X["designation_len"], 50, width=4)
  # st.pyplot(fig)
  buf = BytesIO()
  fig.savefig(buf, format="png")
  st.image(buf)
  
  input = X["designation"].str.split()
  x = [len(element) for element in input]
  sorted_distribution = sorted(x)
  word_count = collections.Counter(sorted_distribution)
  word_count_most_common = word_count.most_common()
  
  '''La répartition de la sémantique des échantillons du champ 'désignation' présente un Pic à **12 Mots**.'''
  fig = plt.figure(figsize=(12, 6))
  plt.hist(x)
  plt.title('Répartition de la sémantique des designations')
  plt.xlabel('Nombre de Mots')
  plt.ylabel('Nombre de Documents')
  plt.xticks([])
  plt.xticks(np.arange(10, 251, 10))
  buf = BytesIO()
  fig.savefig(buf, format="png")
  st.image(buf)
  #st.image(buf)
  #st.pyplot(fig)
  
  
  st.write("#### feature (description)")
  st.write("Répartition de la longueur des descriptions.")
  fig, ax = plt.subplots(figsize = (12,7))
  ax.set_title('Répartition de la longueur des descriptions')
  # fig = plt.figure(figsize=(10, 4))
  ax.set_ylabel("Nb d'occurences")
  ax.set_xlabel('Longueur en caractères')
  plt.xticks(rotation=45)
  plt.hist(X.loc[X.description_len > 0]["description_len"], 50, width=100)
  # st.pyplot(fig)
  buf = BytesIO()
  fig.savefig(buf, format="png")
  st.image(buf)
  
  '''
  ### Analyse des images
  Un script bash est créé pour analyser les images :
  - Elles sont toutes au format JPEG, de taille 500 x 500 pixels.
  - Les tailles de fichiers, en octets, sont enregistrées pour éventuellement enrichir notre dataset.

  Cependant en en visionnant quelques-unes, on constate que beaucoup sont plus petites : elles ont une marge blanche sur 2 ou 4 des côtés. Un script python détermine leur “vraies” dimensions et enrichit notre jeu de données.

  Grâce à ces nouvelles données une analyse de la proportion “utile” des images dans les 500x500 pixels disponibles est effectuée. L’histogramme de cette proportion, comprise entre 0 et 1 et présentée ci-dessous:
  '''
  st.image('distribution-ratio-images.jpg', width=600)
  '''On constate un pic de l’histogramme sur la valeur 1, qui indique qu’une proportion significative des images est de forme carrée ou remplit entièrement les 500x500 pixels disponibles.
  On constate également un pic à une valeur proche de 0.75 qui doit correspondre à un format spécifique d’image très représenté mais qui n’est pas expliqué à ce stade de l’étude.

Une deuxième analyse est effectuée sur le ratio largeur/hauteur de la partie utile des images.
L’histogramme de ces valeurs, ramenées à une échelle log10, montre une distribution proche d’une loi normale, centrée sur 0, qui indique qu’en moyenne la partie utile des images a un aspect carré.
'''
  st.image('distribution-ratio-images-log10.jpg', width=600)
  '''#### Analyse des doublons images

Afin de pouvoir analyser rapidement la présence de doublons d’images, un md5sum des fichiers images est produit et ajouté à la base de données images.

L’analyse de ce md5sum montre que 3264 fichiers images sont au moins présents  à l’identique deux fois dans le jeu de données img_train.

En outre, le maximum de nombre de valeurs distinctes pour chaque md5sum est de 1, l'identifiant produit est donc identique pour les images identiques, le jeu de données est donc cohérent sur ce point.

  ## Enrichissement des données - Modélisation exploitant les meta-data
Dans X_train, ajout des informations suivantes :
- designation_lang : langue détectée par la bibliothèque _langdetect_. Ne semble pas très fiable, rien que dans les 1ère ligne je vois des erreurs.
- description_lang : langue détectée par la bibliothèque _langdetect_
- lang : langue détectée par la bibliothèque _langdetect_ sur la concaténation de designation + description
- designation_len : en nb de caractères
- description_len : en nb de caractères
- image_real_width : en pixels
- image_real_height : en pixels
- image_ratio : largeur / hauteur (width / height)

Le X_train enrichi est stocké dans data/processed.

  ## Enrichissement des données - Modélisation exploitant la sémantique
  Dans X_train, ajout des informations suivantes :
- designation_description : concaténation des champs "designation" et "description"
- lang : langue détectée par la bibliothèque _langdetect_

  La Langue est la **caractéristique** primordiale permettant de capturer la **sémantique**.
  Elle permet notamment d'etre spécifique dans le préprocessing de la donnée textuelle.
  A Noter que la fonction _langdetect_ affiche une **marge d'erreurs** relative d'**~3%** 
  ajoutant une part d'**aléatoire** bien que faible.'''
  '''Le dataset affiche un total de **31 Langues**.'''
  
  '''#### `Liste des langues`'''
  st.write(lang_list)
  
  '''#### `Distribution des langues`'''
  '''La langue **française** est représentée dans plus de **60%** des échantillons et 
  couvre avec la langue **anglaise** **~85%** de la plage des données textuelles.'''
  st.write(lang_dist)




########################################################## Modélisation meta-data ###########################################################
if page == "Modélisation - meta-data" : 
  st.write("## Modélisation sur méta-données")
  '''Les modèles suivants ont été testés :
- LinearSVC
- SGDClassifier
- KNN
- SVC
- GradientBoostingClassifier
- GaussianNB
- XGBoost
- RandomForest

Plusieurs phases ont eu lieu : 
1. Utilisation basique du modèle
2. Ajout de méta-données comme celles provenant de l'OCR (uniquement le nombre de caractères reconnus dans une image)
3. Test de RandomOverSampler RandomUnderSampler
4. Test de GirdSearchCV et de RandomRandomizedSearchCV
5. Unification avec StackingClassifier

Les scores ont généralement augmenté au fur et à mesure, et varient généralement de 10 à 35 %. Ces chiffres sont à comparer avec :
- 3.70 % : l'aléatoire car il y a 27 catégories
- 12.02 % : le poids de la classe la plus importante, si on ne prédisait qu'elle.

TODO : insérer tableau
  '''



########################################################## Modélisation images ###########################################################
if page == "Modélisation - images" : 
  st.write("## Modélisation sur images")




########################################################## Modélisation textes ###########################################################
if page == "Modélisation - textes" : 
  st.write("## Modélisation sur textes")
  '''
- Approche Globale : 
  - Nettoyage des Valeurs **Manquantes**
  - Preprocessing des données textuelles via l'application successive de couches de **_stopwords_** et de couches de **_regex_** 
    tenant compte des **spécificités** de chacune des **langues** et de la **qualité** des données retournées pour chacune d'entre elles
  - **Affinage** Artificiel de la fonction _langdetect_ via l'application d'une **seconde couche** de détection suivie du nettoyage 
    des lignes **non concordantes**
  - **Réequilibrage** des classes par un **sous-echantillonnage** ou un **sur-echantillonnage** du dataset selon la catégorie de langue **_["fr", "en", "others"]_**
    et le modèle de classification retenu
 - Modèles Retenus : **_GradientBoostingClassifier_**, **_Word2Vec_ DNN** Avec **_Keras_**
 - Spécificités du Modèle _GradientBoostingClassifier_ : **Entraînement** du modèle par des sparse matrix csr générées par le vectorisateur **_TfidfVectorizer_** 
 optimisant du processus de **tokenisation/vectorisation** via la prise en compte de la **fréquence d'apparition** de chacun des mots dans un document mais également de son **importance**
 dans le corpus du texte, **Maximisation** de la **performance** du modèle via l'**optimisation** conjointe de son 
 triplet d'hyperparamètres **_{n_estimators, learning_rate, subsample}_**
 - Spécifictiés du Modèle _Word2Vec_ : Réseau **DNN** constitué d'1 couche d'**_Embedding_** en charge de capturer les **relations** entre les mots dans un espace **dense** 
 à dimension **réduite** suivie d'1 couche **_GlobalAveragePooling1D_** permettant la conversion de la matrice de sortie à **2 dimensions** en une matrice à **1 dimension** format requis 
 par la dernière couche **_Dense_** en charge de la classification des **27** catégories, **Entrainement** du Modèle DNN par des matrices de tokens représentant les mots **numérisés** 
 via encodage par le **_one-hot_** et auxquelles a été appliquée au préalable une couche de **padding** via le **_pad_sequences_** afin d'**uniformiser** la longueur des 
 documents du dataset constituant chacun des **batchs** d'entrainement, **Maximisation** de la **performance** du modèle via l'**optimisation** du choix 
 des dimensions de la **matrice des embeddings**, du choix de la **profondeur** du dictionnaire du **vocabulaire**, du choix de l'algorithme 
 de la **descente du gradient** et du choix des paramètres de l'enrainement **_{batch_size, epochs, validation_split}_
 

#### `Script d'optimisation des paramètres d'entraînement du modèle _Word2Vec_`'''
  st.code('''
  batch_size = [64, 128, 256] 
  epochs = [10, 15, 20] 
  param_grid = {"batch_size": [64, 128, 256], "epochs" : [10, 15, 20], "validation_split": [0.1, 0.15, 0.2]}
  param_comb = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
  test_accuracy_list = []
  test_loss_list = []
  for params in param_comb:
            history = model.fit(x_train_fr_pad, y_train_fr_cat, **params)
            test_loss, test_accuracy = model.evaluate(x_test_fr_pad, y_test_fr_cat, verbose = 1)
            test_accuracy_list.append(test_accuracy*100)
            test_loss_list.append(test_loss)''')

  st.code('''best = {}
i = 0
j = 0
max_accuracy = 0
min_loss = 10
for position, accuracy in enumerate(test_accuracy_list):
  if accuracy > max_accuracy:
    max_accuracy = accuracy
    i = position
for position, loss in enumerate(test_loss_list):
  if loss < min_loss:
    min_loss = loss
    j = position
best["best_accuracy_index"] = i
best["best_accuracy_param"] = param_comb[i]
best["best_accuracy"] = max_accuracy
best["best_loss_index"] = j
best["best_loss_param"] = param_comb[j]
best["best_loss"] = min_loss''')





########################################################## Démonstration ###########################################################
if page == "Démonstration" : 
  st.write("## Démonstration")


########################################################## Conclusion ###########################################################
if page == "Conclusion" : 
  st.write("## Conclusion")



########################################################## Page de test ###########################################################
if page == "Page de test" : 
  st.write("## Page de test")
  st.write("# titre 1")
  st.write("## titre 2")
  st.write("### titre 3")
  st.write("#### titre 4")
  st.write("##### titre 5")
  st.write("###### titre 6")
  '''
  Du texte juste mis entre guillemets. Ici *entre astérisques*.
  - des tirets
  - encore des tirets

  Du code python :
  '''
  st.code(''' import streamlit ''', language='python')

# st.write("2025-07-29 1125")
