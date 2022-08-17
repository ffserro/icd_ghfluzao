# icd_ghfluzao
Repositório para apoio às aulas de Introdução a Ciência de Dados

REGRESSÃO LINEAR GHFLUSÃO


from sklearn.datasets import load_boston # para carregar os dados 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # importa o modelo 
# carrega os dados
house_data = load_boston()
X = house_data['data']
y = house_data['target']




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


regr = LinearRegression() # cria o modelo 
regr.fit(X_train, y_train) # treina o modelo


r2_train = regr.score(X_train, y_train)
r2_test = regr.score(X_test, y_test)
print('R2 no set de treino: %.2f' % r2_train)
print('R2 no set de teste: %.2f' % r2_test)




KNN GHFLUSÃO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris() 

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
