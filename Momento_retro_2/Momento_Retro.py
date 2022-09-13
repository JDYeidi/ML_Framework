#Importamos las librerías
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#Leyendo datos de prueba y entrenamiento
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

#2.1 Verificar la cantidad de datos que hay en el dataset
#print(df_test.shape)
#print(df_train.shape)

#2.2 Tipos de datos con los que cuenta el dataset
#print(df_train.info())
#print(df_test.info())

#2.3 Datos faltantes
#print(pd.isnull(df_train).sum())
#print("**************************")
#print(pd.isnull(df_test).sum())

#2.4 Estadisticas de cada dataset
#print(df_test.describe())
#print("**************************")
#print(df_train.describe())

#Cambio de los sexos a numero Label encoding
df_train['Sex'].replace(['female','male'],[0,1], inplace = True)
df_test['Sex'].replace(['female','male'],[0,1], inplace = True)  

#Cambio columna de embarque label encoding
df_train['Embarked'].replace(['Q','S','C'],[0,1,2], inplace = True)
df_test['Embarked'].replace(['Q','S','C'],[0,1,2], inplace = True)

#Rellenando datos faltantes en la columna Age
#print(df_train['Age'].mean())
#print(df_test['Age'].mean())

promedio = 30

df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)

#Creando bandas de edades
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)

#Se elimina cabina porque tiene muchos datos faltantes
df_train.drop(['Cabin'], axis = 1, inplace = True)
df_test.drop(['Cabin'], axis = 1, inplace = True)

#Se eliminan las columnas que no afectan a la predicción final
df_train.drop(['PassengerId', 'Name', 'Ticket'], axis = 1, inplace = True)
df_test.drop(['Name', 'Ticket'], axis = 1, inplace = True)

#Se eliminan las filas restantes ya que son muy pocos datos faltantes
df_train.dropna(axis = 0, how = 'any', inplace = True)
df_test.dropna(axis = 0, how = 'any', inplace = True)

#Separamos la columna que será nuestra variable predictiva
X = np.array(df_train.drop(['Survived'],1))
y = np.array(df_train['Survived'])

#Separamos los datos de manera aleatoria
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Regresion logística
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("Precision con regresión logística: ")
print(logreg.score(X_train, y_train))

#Maquina de soporte vectorial
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Precision MSV: ")
print(svc.score(X_train, y_train))

#K neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Precision Knn: ")
print(knn.score(X_train, y_train))