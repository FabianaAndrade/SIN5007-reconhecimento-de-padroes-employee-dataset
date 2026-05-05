import pandas as pd
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

class NaiveBayes:
    def __init__(self, csv_path, model=GaussianNB(), dataset_name=''):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.X = self.df.drop(columns=['LeaveOrNot'])
        self.y = self.df['LeaveOrNot']
        self.model = model
        self.dataset_name = dataset_name

    def save_report_as_image(self, report, title, filename, accuracy):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-5]:
            row_data = [value for value in line.split(' ') if value]
            if len(row_data) > 1:
                report_data.append(row_data)

        accuracy_line = [value for value in lines[-2].split(' ') if value]
        if accuracy_line:
            report_data.append([accuracy_line[0], '', '', accuracy_line[1], accuracy_line[2]])

        col_labels = ['class', 'precision', 'recall', 'f1-score', 'support']
        
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_title(f"{title}\nAccuracy: {accuracy:.4f}", fontsize=16, pad=20)
        ax.axis('off')
        
        table = ax.table(cellText=report_data, colLabels=col_labels, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def train_test_split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)
        

    def run_train(self):
        self.train_test_split()
    
        model = self.model
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)

        print("\n[MÉTRICAS PRINCIPAIS]")
        print(f"  Acurácia:   {accuracy:.4f}")
        print(f"  Precisão:   {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1-Score:   {f1:.4f}")


        print("Classification Report:\n", report)

        report_title = f'Classification Report - {self.dataset_name} - {type(self.model).__name__}'
        filename = f'figures/classification_report_{self.dataset_name}_{type(self.model).__name__}.png'
        self.save_report_as_image(report, report_title, filename, accuracy)

        #plot confusion matrix
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(self.y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {self.dataset_name} - {type(self.model).__name__}')
        plt.colorbar()
        tick_marks = np.arange(len(set(self.y)))
        plt.xticks(tick_marks, set(self.y), rotation=45)
        plt.yticks(tick_marks, set(self.y))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(f'figures/confusion_matrix_{self.dataset_name}_{type(self.model).__name__}.png', bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":

    csvs = {
        "full": "data/Employee_processed.csv",
        "pca": "data/Employee_pca.csv", #dataset com pca aplicado
        "fs": "data/Employee_fs.csv" #dataset com feature selection aplicado #TODO revisar esse metodo
    }

    
    for key, csv_full_ds in csvs.items():
        print(f"Processando dataset '{key}'...")
        analyzer = NaiveBayes(csv_full_ds, dataset_name=key)
        analyzer.run_train()
        

