import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix

# ler arquivo CSV
datainput = pd.read_csv("bd_diabetes.csv", delimiter=",")

# selecionar as colunas de entrada
X = datainput[['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']].values

# selecionar a coluna de saída (rótulo)
y = datainput["Diabetes_binary"]

# divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# treinamento do modelo
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# teste do modelo
previsoes = modelo.predict(X_test)

# avaliação do modelo
print("\nAcurácia:", accuracy_score(y_test, previsoes),"\n")
print("Matriz de confusão:\n", confusion_matrix(y_test, previsoes))
cm = ConfusionMatrix(modelo)
cm.fit(X_train, y_train)
cm.score(X_test, y_test)

cm.poof()

# plotar o relatório de classificação
report = classification_report(y_test, previsoes, output_dict=True)
df = pd.DataFrame(report).transpose()
df.drop('support', axis=1, inplace=True)
df.plot(kind='bar', rot=0)
plt.title('Relatório de Classificação')
plt.xlabel('Classes')
plt.ylabel('Pontuação')
plt.show()

print("-------------------------------------------------------------")
print(classification_report(y_test, previsoes))


