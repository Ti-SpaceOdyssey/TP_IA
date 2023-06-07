from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score

class ArvoreDecisao:
    def __init__(self, filename):
        self.filename = filename

    def lerDados(self, X_resampled, y_resampled):
        print("\nARVORE DE DECISAO")
        # Inicializa o KFold
        clf = DecisionTreeClassifier(max_depth=5)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # Calcula o valor de validação cruzada
        scores = cross_val_score(clf, X_resampled, y_resampled, cv=kf)
        # Imprime as pontuações de validação cruzada
        print("\nPontuações de Validação Cruzada:", scores)
        print("Média das Pontuações de Validação Cruzada:", np.mean(scores))
        print("Desvio Padrão das Pontuações de Validação Cruzada:", np.std(scores))
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
        decision_rules = []  

        # Itera sobre cada fold
        for train_index, test_index in kf.split(X_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]

            # Treinamento do modelo
            modelo =  DecisionTreeClassifier(max_depth=5)
            modelo.fit(X_train, y_train)

            # Teste do modelo
            previsoes = modelo.predict(X_test)

            # Armazena as métricas deste fold
            accuracies.append(accuracy_score(y_test, previsoes))
            precisions.append(precision_score(y_test, previsoes, average='macro'))
            recalls.append(recall_score(y_test, previsoes, average='macro'))
            f1s.append(f1_score(y_test, previsoes, average='macro'))

            # Armazena as métricas deste fold para classe 0
            precisions_class0.append(precision_score(y_test, previsoes, pos_label=0))
            recalls_class0.append(recall_score(y_test, previsoes, pos_label=0))
            f1s_class0.append(f1_score(y_test, previsoes, pos_label=0))

            # Armazena as métricas deste fold para classe 1
            precisions_class1.append(precision_score(y_test, previsoes, pos_label=1))
            recalls_class1.append(recall_score(y_test, previsoes, pos_label=1))
            f1s_class1.append(f1_score(y_test, previsoes, pos_label=1))

            # Armazena as regras de decisão deste fold
            decision_rules.append(modelo)

        # Imprime as métricas médias

        print("Resultados gerais:")
        print("Acurácia Média:", np.mean(accuracies))
        print("Precisão Média:", np.mean(precisions))
        print("Recall Médio:", np.mean(recalls))
        print("F1-Score Médio:", np.mean(f1s))

        print("\nResultados classe 0:")
        print("Precisão Classe 0 Média:", np.mean(precisions_class0))
        print("Recall Classe 0 Médio:", np.mean(recalls_class0))
        print("F1-Score Classe 0 Médio:", np.mean(f1s_class0))

        print("\nResultados classe 1:")
        print("Precisão Classe 1 Média:", np.mean(precisions_class1))
        print("Recall Classe 1 Médio:", np.mean(recalls_class1))
        print("F1-Score Classe 1 Médio:", np.mean(f1s_class1))

         # Calcula o desvio padrão das métricas
        print("\nDesvio Padrão:")
        print("Desvio Padrão da Acurácia:", np.std(accuracies))
        print("Desvio Padrão da Precisão:", np.std(precisions))
        print("Desvio Padrão do Recall:", np.std(recalls))
        print("Desvio Padrão do F1-Score:", np.std(f1s))

        print("\nDesvio Padrão da Precisão da Classe 0:", np.std(precisions_class0))
        print("Desvio Padrão do Recall da Classe 0:", np.std(recalls_class0))
        print("Desvio Padrão do F1-Score da Classe 0:", np.std(f1s_class0))

        print("\nDesvio Padrão da Precisão da Classe 1:", np.std(precisions_class1))
        print("Desvio Padrão do Recall da Classe 1:", np.std(recalls_class1))
        print("Desvio Padrão do F1-Score da Classe 1:", np.std(f1s_class1))

        # Escreve as regras no arquivo de texto
        with open("regrasArvore.txt", "w") as file:
            for i, modelo in enumerate(decision_rules):
                fold_title = f"\nRegras de decisao para o fold {i+1}:"
                dot_data = export_graphviz(
                    modelo,
                    out_file=None,
                    feature_names=[
                        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
                        'Income'
                    ],
                    class_names=['Nao Diabetico', 'Diabetico'],
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                file.write(fold_title + "\n")
                file.write(dot_data)
                file.write("\n---------------------------\n")