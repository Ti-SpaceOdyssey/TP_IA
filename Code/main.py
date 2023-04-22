from arvoreDecisao import ArvoreDecisao
from diabetesNBComPreProcessamento import NB
import matplotlib.pyplot as plt
import pandas as pd

arvore = ArvoreDecisao("bd_diabetes.csv")
report_tree = arvore.lerDados()

naive = NB("bd_diabetes.csv")
report_naive = naive.lerDados()

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


