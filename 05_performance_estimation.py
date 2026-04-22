from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler # Scalers da aula 04
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import os

# == GRID EXPERIMENT WRAPPER

class GridExperiment:

    def __init__(self, grid):
        
        self.grid = grid

    def grid_run(self, features, target):
        grid.fit(features, target)

    def cv_run(self, X, y, cross_validation_set, scoring_metrics):
        return cross_validate(\
                              self.grid, \
                              X, y, \
                              cv=cross_validation_set, \
                              scoring=scoring_metrics, \
                              verbose=3 \
                              )
        
    def produceGraphics(self):
        pass #vamos acrescentar depois o método para gerar os gráficos de comparação
    

if __name__ == "__main__":

    # Carregando e codificando o CSV
    df = pd.read_csv("data/Employee.csv")

    # Só um copia e cola da aula 02
    # -- Educação
    education_map = {'Bachelors': 1, 'Masters': 2, 'PHD': 3}
    df['Education'] = df['Education'].map(education_map)
    # -- One-hot
    columns_to_encode = ['City', 'Gender', 'EverBenched']
    df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True, dtype=int)

    # Separa dados entre características e alvo
    X = df.drop(columns=['LeaveOrNot'])
    y = df['LeaveOrNot']
    
    pipe = Pipeline([
        ("scaler", "passthrough"),
        ("dimension_reducer", "passthrough"),
        ("classifier", "passthrough"),
    ])

    parameter_grid = [
        {"scaler": [MinMaxScaler()],
         "dimension_reducer": [None, PCA(), SelectKBest(f_classif, k=5)],
         #"dimension_reducer__n_components": [None, 'mle'], # <- Removido por dar problema nos outros redutores de dimensão
         "classifier": [GaussianNB()],
        }
    ]

    # Eu não sei se temos nomes muito melhores que esses
    # Mas, comparando com a aula, até que faz sentido
    cross_validator_externo = StratifiedKFold(n_splits=10, shuffle=True)

    cross_validator_interno = StratifiedKFold(n_splits=5, shuffle=True)
    
    grid = GridSearchCV( \
                        pipe, \
                        parameter_grid, \
                        cv=cross_validator_interno, \
                        verbose=3, \
                        scoring='f1', \
                        refit='f1'
                        )

    scoring_metrics = {
        'f1': 'f1',
        'accuracy': 'accuracy',
        'recall': 'recall',
        'precision': 'precision'
    }
    
    experiment = GridExperiment(grid)

    #experiment.grid_run(X, y)

    data = experiment.cv_run(X, y, cross_validator_externo, scoring_metrics)

    for metric in ['test_f1', 'test_accuracy', 'test_recall', 'test_precision']:
        scores = data[metric]
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        print(f"{metric.replace('test_', '').capitalize()}: {mean_score:.4f} +/- {std_dev:.4f}")
    
    experiment.produceGraphics()
