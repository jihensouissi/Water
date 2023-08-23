# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Basic Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from warnings import filterwarnings
from collections import Counter

# Visualizations Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objs as go

import plotly.figure_factory as ff
import missingno as msno

# Data Pre-processing Libraries
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split

# Modelling Libraries
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import VotingClassifier

# Evaluation & CV Libraries
from sklearn.metrics import precision_score,accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,RepeatedStratifiedKFold




import pandas as pd

# -

# !pip install missingno

# # About Database

# +

#df = pd.read_csv(r"C:\Users\asus tuf\Downloads\seawater.csv")
df = pd.read_csv(r"C:\Users\asus tuf\Downloads\seawater.csv")

# +

df.head()
# -

print(df.shape)

df.info()

print(df.columns)

df.describe()

# number of unique values in each column 
print(df.nunique())

df.dtypes

unique_values =df['status'].unique()
print(unique_values)

ax = sns.countplot(x = "status",data= df, saturation=0.8)
plt.xticks(ticks=[0, 1], labels = ["Not APTA", "APTA"])
plt.show()

df['id_site'].unique()

unique_site_count = len(df['id_site'].unique())
print(unique_site_count)


# # Data Preparation


df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

df['Timestamp'] = df['date'].apply(lambda x: x.timestamp())


data_type = df['Timestamp'].dtype
print(data_type)

del df['date']

# +
import pandas as pd

# Assuming your dataset is stored in a DataFrame named df and the "hour" column contains time values in string format
df['hour'] = pd.to_datetime(df['hour'])  # Convert the "hour" column to datetime type
df['Hour'] = df['hour'].dt.hour + df['hour'].dt.minute / 60 + df['hour'].dt.second / 3600


# -

data_type = df['Hour'].dtype
print(data_type)

del df["hour"]

del df["timestamp_sampled"]

# +

missing_values = df.isnull().sum()
print(missing_values)

# -

unique_values =df['water_temp'].unique()
print(unique_values)

data_type = df['water_temp'].dtype
print(data_type)

# +

df['water_temp'] = pd.to_numeric(df['water_temp'], errors='coerce')
df['dis_oxig'] = pd.to_numeric(df['dis_oxig'], errors='coerce')
df['water_ph'] = pd.to_numeric(df['water_ph'], errors='coerce')

# -

df['water_salinity'] = pd.to_numeric(df['water_salinity'], errors='coerce')
data_type = df['water_salinity'].dtype
print(data_type)

unique_values =df['status'].unique()
print(unique_values)

df["status"].value_counts()

df['status'] = df['status'].map({'APTA': 1, 'NO APTA': 0})

unique_values =df['status'].unique()
print(unique_values)

colors_blue = ["#132C33", "#264D58", '#17869E', '#51C4D3', '#B4DBE9']
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_green = ['#01411C','#4B6F44','#4F7942','#74C365','#D0F0C0']


# +

# -

df['id_site'].unique()

ids=df['id_site'].unique()
len(ids)

df['id_site'].replace(ids,range(1,25),inplace=True)
df['id_site']

del df['site_name']


types=df['site_type'].unique()


# +

df['site_type'].replace(types,range(1,len(types)+1),inplace=True)
df['site_type']
# -

print(df.dtypes)





# +
#int to float
df['id_site'] = df['id_site'].astype(float)
df['site_type'] = df['site_type'].astype(float)

df['status'] = df['status'].astype(float)
# -

df.head()

df.describe()

# +
# Configuracion de base del histograma

fig = px.histogram(df, x="water_ph", nbins=70, color='status', template='plotly_white', 
                   marginal='box', opacity=0.7, color_discrete_sequence=[colors_green[3],colors_blue[3]],
                   histfunc='count')

# linea en area pH neutro valor de 7
fig.add_vline(x=7, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

# ajuste para centrar histograma e indicar cuando los valores son acidicos y basicos
fig.add_annotation(text='<7 is Acidic',x=5.5,y=70,showarrow=False,font_size=12)
fig.add_annotation(text='>7 is Basic',x=10,y=70,showarrow=False,font_size=12)
fig.add_annotation(text='1 = Apt',x=10,y=300,showarrow=False,font_size=11)
fig.add_annotation(text='0 = Not Apt',x=10,y=280,showarrow=False,font_size=11)

# Titulos y etiquetas en 
fig.update_layout(
    font_family='monospace',
    title=dict(text='pH Level Distribution at Bahia de Loreto National Park',x=0.5,y=0.95,
               font=dict(color=colors_dark[2],size=20)),
    xaxis_title_text='pH Level',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
    bargap=0.3,
)

fig.show()

# +
# Configuracion de base del histograma
fig = px.histogram(df, x="dis_oxig", nbins=50, color='status', template='plotly_white', 
                   marginal='box', opacity=0.7, color_discrete_sequence=[colors_green[3],colors_blue[3]],
                   histfunc='count')

                                   

# Reqerimientos oxigeno por especie
fig.add_annotation(text='menor 3.7<br> vida marina<br> abandonan area',x=3.3,y=500,showarrow=False,font_size=12)
fig.add_vline(x=3.8, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)
fig.add_annotation(text='la vida marina<br> evita este rango',x=4.5,y=500,showarrow=False,font_size=12)
fig.add_vline(x=5.5, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)
fig.add_annotation(text='este rango es<br> adecuado para la vida marina',x=6.9,y=500,showarrow=False,font_size=12)
fig.add_vline(x=8, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)
fig.add_annotation(text='rango de<br> saturacion al 100%',x=8.9,y=500,showarrow=False,font_size=12)

fig.add_annotation(text='1 = Apt',x=10,y=880,showarrow=False,font_size=11)
fig.add_annotation(text='0 = Not Apt',x=10,y=840,showarrow=False,font_size=11)



# Titulos y etiquetas en 
fig.update_layout(
    font_family='monospace',
    title=dict(text='Dissolved Oxygen Distribution at the Bahia de Loreto National Park',x=0.5,y=0.95,
               font=dict(color=colors_dark[2],size=20)),
    xaxis_title_text='DO Level in mg/L',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
    bargap=0.3,
)

fig.show()

# +
# Configuracion de base del histograma
fig = px.histogram(df, x="water_salinity", nbins=50, color='status', template='plotly_white', 
                   marginal='box', opacity=0.7, color_discrete_sequence=[colors_green[3],colors_blue[3]],
                   histfunc='count')

# linea valor medio
fig.add_vline(x=35, line_width=1, line_color=colors_dark[1],line_dash='dot',opacity=0.7)

# ajuste para centrar histograma e indicar cuando los valores son aptos y no aptos
fig.add_annotation(text='1 = Apt',x=38,y=1000,showarrow=False,font_size=11)
fig.add_annotation(text='0 = Not Apt',x=38,y=900,showarrow=False,font_size=11)

# Titulos y etiquetas en 
fig.update_layout(
    font_family='monospace',
    title=dict(text='Salinity Level Distribution at Bahia de Loreto National Park',x=0.5,y=0.95,
               font=dict(color=colors_dark[2],size=20)),
    xaxis_title_text='Salinity Level in ppt',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,bordercolor=colors_dark[4],borderwidth=0,tracegroupgap=5),
    bargap=0.3,
)

fig.show()
# -

# # Dealing with outliers

#Detecting and Treating Outliers
outliers=[]
def detect_outliers(feature):
    threshold=10
    mean=np.mean(feature) #moyenne
    std=np.std(feature) # Compute the standard deviation of the given data
    for i in feature:
        z_score=(i-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(i)
    return(outliers)


# +
red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, axs = plt.subplots(1, len(df.columns), figsize=(40,10))

for i, ax in enumerate(axs.flat):
    ax.boxplot(df.iloc[:,i], flierprops=red_circle)
    ax.set_title(df.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    
plt.tight_layout()
# -

outlier_en=detect_outliers(df['enterococoe'])
outlier_en



df=df[df.enterococoe <14370.0]
df.shape

outlier_l2=detect_outliers(df['latitude'])
outlier_l2

df=df[df.latitude <24196.0]
df.shape

outlier_l3=detect_outliers(df['longitude'])
outlier_l3

df=df[df.longitude <14370.0]
df.shape

cleaned_df = df.dropna()

cleaned_df.duplicated().sum()

#











# +
import pandas as pd

# Assume you have performed data cleaning and the cleaned DataFrame is named 'cleaned_data'
# For example:
# cleaned_data = ... (your cleaned DataFrame)

# Save the cleaned data to a CSV file
cleaned_df.to_csv('data-seaWater.csv', index=False)

# The CSV file will be saved in the current working directory of your Jupyter Notebook

# To download the file from Jupyter Notebook, you can use the following code:

from IPython.display import FileLink

# Provide the filename along with the path if it's not in the current working directory
file_path = 'data-seaWater.csv'

# Create a FileLink object and display it
FileLink(file_path)


# +

#df = pd.read_csv(r"C:\Users\asus tuf\Downloads\seawater.csv")
data = pd.read_csv(r"data-seaWater.csv")
# -

data.dtypes

#
# # Deal with imbalanced datasets

# +
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Séparez vos données en caractéristiques (X) et étiquettes de classe (y)
X = data.drop("status", axis=1)  # Assurez-vous d'exclure la colonne "status"
y = data["status"]


# Créez une instance de la technique de sur-échantillonnage SMOTE
smote = SMOTEENN(sampling_strategy='minority', random_state=1000)


# Appliquez le sur-échantillonnage sur vos données
X_resampled, y_resampled = smote.fit_resample(X, y)

# -

smote

X_resampled

y_resampled

# +
X_resampled['status'] = y_resampled

# Affichez le DataFrame mis à jour
print(X_resampled)


# +
import pandas as pd
# Save the  data to a CSV file
X_resampled.to_csv('data-seaWater2.csv', index=False)

# The CSV file will be saved in the current working directory of your Jupyter Notebook

# To download the file from Jupyter Notebook, you can use the following code:

from IPython.display import FileLink

# Provide the filename along with the path if it's not in the current working directory
file_path = 'data-seaWater2.csv'

# Create a FileLink object and display it
FileLink(file_path)

# -

data2 = pd.read_csv(r"data-seaWater2.csv")

data2

ax = sns.countplot(x = "status",data= data2, saturation=0.8)
plt.xticks(ticks=[0, 1], labels = ["0.0", "1.0"])
plt.show()

unique_counts = data2["status"].value_counts().unique()
unique_counts

# #  feature Selection

# +
import pandas as pd
import numpy as np
import seaborn as sns

#get correlations of each features in dataset
corrmat = data2.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# -

# Séparez vos données en caractéristiques (X) et étiquettes de classe (y)
X = data2.drop("status", axis=1)  # Assurez-vous d'exclure la colonne "status"
y = data2["status"]

# +

#X = df.iloc[:,0:29]  #independent columns
#y = df.iloc[:,-1]    #target column i.e price range
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(25).plot(kind='barh')
plt.title("Features importances")
plt.show()
# -





from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
rfe= RFE(estimator =DecisionTreeClassifier() , n_features_to_select=7)
rfe.fit(X_resampled,y_resampled)



for i ,col in zip(range(X.shape[1]),data2.columns):
    print(f"{col} selected ={rfe.support_[i]} rank={rfe.ranking_[i]}")

# +
del data2['id_site']


# -



del data2['site_type']
del data2['latitude']
del data2['longitude']

# # Modelisation



from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_friedman1
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.svm import SVC

# +
from sklearn.model_selection import train_test_split

#X = data.drop('status', axis=1)
#Y = data['status']
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
X1 = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.25, random_state=42)

# +
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


n_neighbors = list(range(1,100))
p=[1,2] #p = 1 manhattan_distance/p=2 euclidean_distance
#Convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsRegressor()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=15)

#cv: number of folds of the cross validation

#Fit the model
best_model = clf.fit(X_train, y_train)
#Print The value of best Hyperparameters
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
# -

# Afficher les paramètres qui donnent les meilleurs performances
best_parameters = clf.best_params_
print(best_parameters)

knn = KNeighborsClassifier(1,p=1)
knn_model = knn.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))

# The training-set accuracy score is 1 while the test-set accuracy to be 0.9700. These two values are quite comparable. So, there is no sign of overfitting

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn))

# +
#avec cross validation

# +
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Supposons que vous avez déjà divisé vos données en X_train et y_train

# Créez une instance du classifieur KNN
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # Vous pouvez ajuster le nombre de voisins (n_neighbors) selon vos besoins

# Effectuez la validation croisée avec 5 folds (plis)
num_folds = 5
cross_val_scores = cross_val_score(knn_classifier, X_train, y_train, cv=num_folds, scoring='accuracy')

# Les scores de validation croisée pour chaque pli
print("Cross-validation scores:", cross_val_scores)

# Calcul de la moyenne des scores de validation croisée
mean_cv_score = cross_val_scores.mean()
print("Mean cross-validation score:", mean_cv_score)

# -



# +
# visualize confusion matrix with seaborn heatmap
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_knn)

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# +
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for KNN Classifier for Predicting Quality')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

# +
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred_knn)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
# -

# # logistic reggression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


print('Accuracy of logistic Regression classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of logistic Regression classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_pred)))

# Print confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# +
# visualize confusion matrix with seaborn heatmap
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

# +
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Logistic regression Classifier for Predicting Quality')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

# +
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
# -

# # SVM

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# +
import psutil  # Pour surveiller l'utilisation de la mémoire
import time   # Pour mesurer le temps d'exécution

# Mesurer l'utilisation de la mémoire avant l'entraînement
memory_before = psutil.virtual_memory().used

# +
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Entraîner le modèle et mesurer le temps d'exécution
start_time = time.time()

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
modelSVM =grid.fit(X_train_scaled, y_train)

end_time = time.time()


# +
# Mesurer l'utilisation de la mémoire après l'entraînement
memory_after = psutil.virtual_memory().used

# Calculer le temps d'exécution en secondes
execution_time = end_time - start_time

# Estimation de la consommation de mémoire en bytes
memory_consumption = memory_after - memory_before

# Afficher les résultats
print("Temps d'exécution SVM :", execution_time, "secondes")
print("Consommation de mémoire du modéle SVM:", memory_consumption, "octets")
# -

y_pred2 = grid.predict(X_test_scaled)


# +
# visualize confusion matrix with seaborn heatmap
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred2)

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
# -

# Print confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("\nClassification Report:\n", classification_report(y_test, y_pred2))


# +
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred2)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for SVM Classifier for Predicting Quality')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

# +
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_pred2)

print('ROC AUC : {:.4f}'.format(ROC_AUC))
# -




# # Random FOREST

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# +
import psutil  # Pour surveiller l'utilisation de la mémoire
import time   # Pour mesurer le temps d'exécution

# Mesurer l'utilisation de la mémoire avant l'entraînement
memory_beforeF = psutil.virtual_memory().used

# +
param_grid2 = {
    'n_estimators':  [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Entraîner le modèle et mesurer le temps d'exécution
start_timeF = time.time()


grid2 = GridSearchCV(RandomForestClassifier(random_state=42), param_grid2, refit=True, verbose=3)
grid2.fit(X_train, y_train)
end_timeF = time.time()


# +
# Mesurer l'utilisation de la mémoire après l'entraînement
memory_afterF = psutil.virtual_memory().used

# Calculer le temps d'exécution en secondes
execution_timeF = end_timeF - start_timeF

# Estimation de la consommation de mémoire en bytes
memory_consumptionF = memory_afterF - memory_beforeF

# Afficher les résultats
print("Temps d'exécution dde RF:", execution_timeF, "secondes")
print("Consommation de mémoire de modéle RF:", memory_consumptionF, "octets")
# -

y_predF = grid2.predict(X_test)


# +
# visualize confusion matrix with seaborn heatmap
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predF)

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
# -

# Print confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_predF))
print("\nClassification Report:\n", classification_report(y_test, y_predF))


# +
# plot ROC Curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_predF)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 12

plt.title('ROC curve for Random Forest Classifier for Predicting Quality')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

# +
# compute ROC AUC

from sklearn.metrics import roc_auc_score

ROC_AUC = roc_auc_score(y_test, y_predF)

print('ROC AUC : {:.4f}'.format(ROC_AUC))

# +
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset and split it into features (X) and labels (y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier with different maximum depths
max_depths = [None, 5, 10, 20]
for max_depth in max_depths:
    model = DecisionTreeClassifier(max_depth=max_depth)
    
    # Train the model on the training set
    model.fit(X_train, y_train)
    
    # Predict on the training set and calculate accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
    # Predict on the validation/test set and calculate accuracy
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"Max Depth: {max_depth}")
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Validation/Test Accuracy: {test_accuracy:.2f}")
    print()

# -

# # deploiement

# +
import joblib  # Pour sauvegarder avec joblib

# Sauvegarder le modèle entraîné
joblib.dump(modelSVM, 'svm_model.pkl')

# -



# +
import joblib

# Charger le modèle depuis le fichier
loaded_model = joblib.load('svm_model.pkl')

# Utiliser le modèle chargé pour faire des prédictions
predictions = loaded_model.predict(X_train)

# -

predictions

# !pip install skl2onnx



# +
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Specify initial types for the model inputs
initial_type = [('input', FloatTensorType([None, X.shape[1]]))]

# Convert the scikit-learn model to ONNX format
onnx_model = convert_sklearn(loaded_model, "SVM Model", initial_types=initial_type)

# Save the ONNX model to a file
onnx_filename = "svm_model.onnx"
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())

# -

#onnx_filename = "C:\Users\asus tuf\Desktop\seaWater\svm_model.onnx"




# !pip install streamlit
