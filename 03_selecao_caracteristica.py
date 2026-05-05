import pandas as pd
import seaborn as sns
from ReliefF import ReliefF
import matplotlib.pyplot as plt
import numpy as np



#https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83

class SelecaoCaracteristica:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.X = self.df.drop(columns=['LeaveOrNot'])
        self.y = self.df['LeaveOrNot']

    def reliefF(self, colunas_drop):
        self.X = self.X.drop(columns=colunas_drop)
        n_features_to_keep = self.X.shape[1]
        print("Running ReliefF with n_features_to_keep =", n_features_to_keep)
        fs = ReliefF(n_neighbors=1, n_features_to_keep=n_features_to_keep)
        X_train = fs.fit_transform(self.X.values, self.y.values)
        print("Transformed data:")
        print(X_train)
        print("--------------")
        print("Original shape: "+str(self.df.shape))
        print("Transformed shape: "+str(X_train.shape))
        print("--------------")
        print("Selected feature names:")
        print(self.X.columns[fs.top_features])

        selected_feature_names = self.X.columns[fs.top_features[:fs.n_features_to_keep]]
        df = pd.DataFrame(X_train, columns=selected_feature_names)
        df['LeaveOrNot'] = self.y.values

        df.to_csv("data/Employee_fs.csv", index=False)
        

    def plot_feature_importance(self):
        fs = ReliefF(n_neighbors=1, n_features_to_keep=self.X.shape[1])
        fs.fit(self.X.values, self.y.values)

        feature_scores = fs.feature_scores
        feature_names = self.X.columns

        sorted_indices = np.argsort(feature_scores)[::-1]
        sorted_scores = feature_scores[sorted_indices]
        sorted_names = feature_names[sorted_indices]
        return sorted_names, sorted_scores

    def save_scores_as_image(self, names, scores, filename):
        data = [[name, f"{score:.1f}"] for name, score in zip(names, scores)]
        col_labels = ['Feature', 'Score']
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title('Tabela 4 - Feature Importance Scores', fontsize=16, pad=20)
        ax.axis('off')
        
        table = ax.table(cellText=data, colLabels=col_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def run_analysis(self):
        
        feature_names, feature_scores = self.plot_feature_importance()
        print("Feature names sorted by importance:")
        for name, score in zip(feature_names, feature_scores):
            print(f"{name}: {score:.4f}")

        self.save_scores_as_image(feature_names, feature_scores, 'figures/feature_importance_scores.png')
        
        self.reliefF(['Age'])


if __name__ == "__main__":
    analyzer = SelecaoCaracteristica("data/Employee_processed.csv")
    analyzer.run_analysis()