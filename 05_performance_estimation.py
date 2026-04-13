from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler # Scalers da aula 04
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import os

# == GRID EXPERIMENT WRAPPER

class GridExperiment:

    def __init__(self, grid):

        self.grid = grid

    def run(self, features, target):
        grid.fit(features, target)
        
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
         "dimension_reducer": [PCA()],
         "dimension_reducer__n_components": [None, 'mle'],
         "classifier": [GaussianNB()],
        }
    ]

    k_folds = KFold(n_splits=5, shuffle=True)
    
    grid = GridSearchCV(pipe, parameter_grid, cv=k_folds, verbose=3, scoring='f1')

    experiment = GridExperiment(grid)

    experiment.run(X, y)
    
    experiment.produceGraphics()
