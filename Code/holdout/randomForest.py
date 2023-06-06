import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from imblearn.under_sampling import RandomUnderSampler


class RandomForests:
    def __init__(self, filename):
        self.filename = filename

    def lerDados(self,X_resampled, y_resampled):
        # divisão dos dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=3)

        # treinamento do modelo
        modelo = RandomForestClassifier(n_estimators=100)
        modelo.fit(X_train, y_train)

        # teste do modelo
        previsoes = modelo.predict(X_test)

        # avaliação do modelo
        print("\nAcurácia:", accuracy_score(y_test, previsoes), "\n")
        print("Matriz de confusão:\n", confusion_matrix(y_test, previsoes))
        cm = ConfusionMatrix(modelo)
        cm.fit(X_train, y_train)
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

        return report