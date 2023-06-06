from arvoreDecisao import ArvoreDecisao
from randomForest import RandomForests
from naiveBayes import NB
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Define a função para agrupar as idades


def age_to_group(age):
    if age in range(1, 4):
        return 1
    elif age in range(5, 8):
        return 2
    elif age in range(9, 12):
        return 3
    elif age == 13:
        return 4
    else:
        return 5

# Define a função para agrupar as faixas de renda


def income_to_group(income):
    if income in range(1, 2):
        return 1
    elif income in range(3, 4):
        return 2
    elif income in range(5, 6):
        return 3
    elif income == 7:
        return 4
    else:
        return 5

# Define a função para alterar a logica da classificacao de saude


def genhlth_to_group(genhlth):
    if genhlth == 5:
        return 1
    elif genhlth == 4:
        return 2
    elif genhlth == 3:
        return 3
    elif genhlth == 2:
        return 4
    else:
        return 5


# ler arquivo CSV
datainput = pd.read_csv("bd_diabetes.csv", delimiter=",")

# tratar outliers
for col in datainput.columns:
    if col in ['BMI',  'MentHlth', 'PhysHlth']:
        datainput = datainput[np.abs(
            datainput[col] - datainput[col].mean()) / datainput[col].std() < 3]

datainput = datainput.drop_duplicates()  # eliminar redundancia

# Aplica a função à coluna 'Age' e sobrescreve os valores originais
datainput['Age'] = datainput['Age'].apply(age_to_group)

# Aplica a função à coluna 'Income' e sobrescreve os valores originais
datainput['Income'] = datainput['Income'].apply(income_to_group)

# Aplica a função à coluna 'GenHlth' e sobrescreve os valores originais
datainput['GenHlth'] = datainput['GenHlth'].apply(genhlth_to_group)

# Verificar a matriz de correlação
correlation_matrix = datainput.corr()
# print(correlation_matrix)

# Selecionar as colunas de entrada
input_columns = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
X = datainput[input_columns].values

# selecionar a coluna de saída (rótulo)
y = datainput["Diabetes_binary"]

# undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
num_instances = X_resampled.shape[0]
print("Número total de instâncias após a amostragem:", num_instances)
counts = np.bincount(y_resampled)
total_classe_0 = counts[0]
total_classe_1 = counts[1]
print("Total da classe 0:", total_classe_0)
print("Total da classe 1:", total_classe_1)

print("0: encerrar programa")
print("1: Arvore")
print("2: Naive")
print("3: Random Forest")

escolha = input("digitar escolha: ")

while(int(escolha) != 0):
    if (int(escolha) == 1):
        arvore = ArvoreDecisao("bd_diabetes.csv")
        report_tree = arvore.lerDados(X_resampled, y_resampled)

    elif (int(escolha) == 2):
        naive = NB("bd_diabetes.csv")
        report_naive = naive.lerDados(X_resampled, y_resampled)


    elif (int(escolha) == 3):
        rf = RandomForests("bd_diabetes.csv")
        rf.lerDados(X_resampled, y_resampled, input_columns)
    
    print("--------------------------------------------------------------------------------------\n")
    print("0: encerrar programa")
    print("1: Arvore")
    print("2: Naive")
    print("3: Random Forest")
    escolha = input("digitar escolha: ")
    

