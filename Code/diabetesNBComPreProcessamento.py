import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from imblearn.under_sampling import RandomUnderSampler

# ler arquivo CSV
datainput = pd.read_csv("bd_diabetes.csv", delimiter=",")

# Define a função para agrupar as idades
def age_to_group(age):
    if age in range(18, 40):
        return 1
    elif age in range(40, 60):
        return 2
    elif age in range(60, 80):
        return 3
    elif age >= 80:
        return 4
    else:
        return 5

# Aplica a função à coluna 'Age' e sobrescreve os valores originais
datainput['Age'] = datainput['Age'].apply(age_to_group)

# Define a função para agrupar as faixas de renda
def income_to_group(income):
    if income < 15000:
        return 1
    elif income < 25000:
        return 2
    elif income < 50000:
        return 3
    elif income < 75000:
        return 4
    else:
        return 5

# Aplica a função à coluna 'Income' e sobrescreve os valores originais
datainput['Income'] = datainput['Income'].apply(income_to_group)



# selecionar as colunas de entrada
X = datainput[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']].values


# selecionar a coluna de saída (rótulo)
y = datainput["Diabetes_binary"]


# undersampling
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

# divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=3)

# treinamento do modelo
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# teste do modelo
previsoes = modelo.predict(X_test)

# avaliação do modelo
print("\nAcurácia:", accuracy_score(y_test, previsoes), "\n")
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
