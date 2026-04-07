import pandas as pd
import seaborn as sns
from ReliefF import ReliefF



#https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83

class SelecaoCaracteristica:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.X = self.df.drop(columns=['LeaveOrNot'])
        self.y = self.df['LeaveOrNot']

    def reliefF(self):
        fs = ReliefF(n_neighbors=1, n_features_to_keep=2)
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
        

    def run_analysis(self):
        self.reliefF()


if __name__ == "__main__":
    analyzer = SelecaoCaracteristica("data/Employee_processed.csv")
    analyzer.run_analysis()