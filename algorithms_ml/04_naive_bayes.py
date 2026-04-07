import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NaiveBayes:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.X = self.df.drop(columns=['LeaveOrNot'])
        self.y = self.df['LeaveOrNot']

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        

    def run_train(self):
        self.train_test_split()
    
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))


if __name__ == "__main__":

    csvs = {
        "full": "data/Employee_processed.csv",
        "pca": "data/Employee_pca.csv", #dataset com pca aplicado
        "fs": "data/Employee_fs.csv" #dataset com feature selection aplicado #TODO revisar esse metodo
    }
    
    for key, csv_full_ds in csvs.items():
        print(f"Running Naive Bayes on '{key}' dataset:")

        analyzer = NaiveBayes(csv_full_ds)
        analyzer.run_train()