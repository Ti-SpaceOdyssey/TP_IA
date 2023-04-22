from arvoreDecisao import ArvoreDecisao
from diabetesNBComPreProcessamento import NB
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# ler arquivo CSV
datainput = pd.read_csv("bd_diabetes.csv", delimiter=",")

datainput = datainput.drop_duplicates() #eliminar redundancia

# Define a função para agrupar as idades
def age_to_group(age):
    if age in range(1, 4):
        return 1
    elif age in range(5, 8):
        return 2
    elif age in range(9, 12):
        return 3
    elif age  == 13 :
        return 4
    else:
        return 5


# Aplica a função à coluna 'Age' e sobrescreve os valores originais
datainput['Age'] = datainput['Age'].apply(age_to_group)

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

# Aplica a função à coluna 'Income' e sobrescreve os valores originais
datainput['Income'] = datainput['Income'].apply(income_to_group)

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


# Aplica a função à coluna 'GenHlth' e sobrescreve os valores originais
datainput['GenHlth'] = datainput['GenHlth'].apply(genhlth_to_group)

# Verificar a matriz de correlação
correlation_matrix = datainput.corr()
print(correlation_matrix)

# selecionar as colunas de entrada
X = datainput[['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']].values

# selecionar a coluna de saída (rótulo)
y = datainput["Diabetes_binary"]

# undersampling
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

arvore = ArvoreDecisao("bd_diabetes.csv")
report_tree = arvore.lerDados(X_resampled, y_resampled)

naive = NB("bd_diabetes.csv")
report_naive = naive.lerDados(X_resampled, y_resampled)

# transformar as métricas do relatório em um DataFrame
df_tree = pd.DataFrame(report_tree).transpose()
df_naive = pd.DataFrame(report_naive).transpose()

# remover a coluna 'support'
df_tree.drop('support', axis=1, inplace=True)
df_tree['model'] = 'Decision Tree' # Adicionar coluna "model" ao dataframe
df_naive.drop('support', axis=1, inplace=True)
df_naive['model'] = 'Naive Bayes' # Adicionar coluna "model" ao dataframe



# Concatenar dataframes dos modelos em um único dataframe
df = pd.concat([df_tree, df_naive], axis=0)

# Plotar gráfico de barras com informações dos dois modelos
ax = df.plot(kind='barh', x='model', rot=0)
ax.invert_yaxis()
ax.set_title('Relatório de Classificação')
ax.set_xlabel('Modelos')
ax.set_ylabel('Pontuação')
plt.show()


