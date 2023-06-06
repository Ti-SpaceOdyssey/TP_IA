from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy import stats

class NB:
    def __init__(self, filename):
        self.filename = filename

    def lerDados(self,X_resampled, y_resampled):
        print("\nNAIVE BAYES")
        # Inicializa o KFold
        clf = GaussianNB()
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(clf,X_resampled,y_resampled, cv = kf)
        print("\nPontuações de Validação Cruzada:", scores)
        print("Média das Pontuações de Validação Cruzada:", np.mean(scores))
        print("Desvio Padrão das Pontuações de Validação Cruzada:", np.std(scores))

        # Calculate the critical value for 95% confidence level
        alpha = 0.05
        critical_value = stats.t.ppf(1 - alpha / 2, len(X_resampled) - 1)

        # Print the critical value
        print("Critical Value:", critical_value)

        # Inicializa as listas para armazenar as métricas em cada fold
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        precisions_class0 = []
        recalls_class0 = []
        f1s_class0 = []
        precisions_class1 = []
        recalls_class1 = []
        f1s_class1 = []

        # Itera sobre cada fold
        for train_index, test_index in kf.split(X_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]

            # treinamento do modelo
            modelo = GaussianNB()
            modelo.fit(X_train, y_train)

            # Print the size of the training set
            print("Tamanho do conjunto de treino:", len(X_train))

            # teste do modelo
            previsoes = modelo.predict(X_test)

            # armazena as métricas deste fold
            accuracies.append(accuracy_score(y_test, previsoes))
            precisions.append(precision_score(y_test, previsoes, average='macro'))
            recalls.append(recall_score(y_test, previsoes, average='macro'))
            f1s.append(f1_score(y_test, previsoes, average='macro'))

            # armazena as métricas deste fold para classe 0
            precisions_class0.append(precision_score(y_test, previsoes, pos_label=0))
            recalls_class0.append(recall_score(y_test, previsoes, pos_label=0))
            f1s_class0.append(f1_score(y_test, previsoes, pos_label=0))

            # armazena as métricas deste fold para classe 1
            precisions_class1.append(precision_score(y_test, previsoes, pos_label=1))
            recalls_class1.append(recall_score(y_test, previsoes, pos_label=1))
            f1s_class1.append(f1_score(y_test, previsoes, pos_label=1))

        # Imprime as métricas médias
        
        print("Resultados gerais:")
        print("Acurácias:", accuracies)
        print("Acurácia Média:", np.mean(accuracies))
        print("Precisões:", precisions)
        print("Precisão Média:", np.mean(precisions))
        print("Recall's:", recalls)
        print("Recall Médio:", np.mean(recalls))
        print("F1-Score's:", f1s)
        print("F1-Score Médio:", np.mean(f1s))
        
        print("\nResultados classe 0:")
        print("Precisão Classe 0 Média:", np.mean(precisions_class0))
        print("Recall Classe 0 Médio:", np.mean(recalls_class0))
        print("F1-Score Classe 0 Médio:", np.mean(f1s_class0))
        
        print("\nResultados classe 1:")
        print("Precisão Classe 1 Média:", np.mean(precisions_class1))
        print("Recall Classe 1 Médio:", np.mean(recalls_class1))
        print("F1-Score Classe 1 Médio:", np.mean(f1s_class1))
