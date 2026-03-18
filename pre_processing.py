import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np




class PreProcessing:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def feature_engineering(self):
        current_year = 2018
        self.df['YearsAtCompany'] = current_year - self.df['JoiningYear']
        self.df = self.df.drop(columns=['JoiningYear'])
        return self.df

    def ordinal_encode(self):
        education_map = {'Bachelors': 1, 'Masters': 2, 'PHD': 3}
        self.df['Education'] = self.df['Education'].map(education_map)
        return self.df

    def scale_numerical(self):
        scaler = MinMaxScaler()
        self.df[['Age']] = scaler.fit_transform(self.df[['Age']])
        return self.df

    def one_hot_encode(self):
        columns_to_encode = ['City', 'Gender', 'EverBenched']
        self.df = pd.get_dummies(self.df, columns=columns_to_encode, drop_first=True, dtype=int)
        print("Dataset após o Pré-Processamento:")
        print(self.df.head())
        return self.df
    
    def get_df_basic_info(self):
        print("Dataset shape:", self.df.shape)
        
        summary_data = []
        for col in self.df.columns:
            dtype_obj = self.df[col].dtype
            non_null = self.df[col].count()
            unique = self.df[col].nunique()
            
            if "int" in str(dtype_obj):
                tipo_num = 'Numérico Discreto'
            elif "float" in str(dtype_obj):
                tipo_num = 'Numérico Contínuo'
            else:
                tipo_num = 'Categórico/Outro'
                
            summary_data.append([col, tipo_num, non_null, unique])
        
        fig, ax = plt.subplots(figsize=(12, len(self.df.columns) * 0.5 + 1))
        ax.axis('tight')
        ax.axis('off')
        
        col_labels = ['Coluna', 'Tipo de Dado', 'Non-null Count', 'Valores Únicos']
        table = ax.table(cellText=summary_data, 
                        colLabels=col_labels, 
                        cellLoc='center', 
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5) 
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')

        plt.title('Tabela 3 - Informações do dataset após pré-processamento', y=0.98)
        plt.subplots_adjust(top=0.85)
        
        plt.savefig('figures/tb_basic_info_pos_proc.png', bbox_inches='tight', dpi=300)
        plt.show()
        
        
    def plot_distributions(self):
        cols = self.df.columns
        n_cols = 3
        n_rows = math.ceil(len(cols) / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            
            if self.df[col].nunique() <= 5:
                sns.countplot(data=self.df, x=col, ax=axes[i], palette='gray')
                axes[i].set_title(f'Distribuição de {col}')
            else:
                sns.histplot(self.df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f'Distribuição de {col}')
            
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Frequência')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.suptitle('Figura 6 - Distribuição das Variáveis (Pós-Processamento)', y=1.02, fontsize=16)
        plt.savefig('figures/distribuicoes_variaveis_pos_proc.png', bbox_inches='tight')
        plt.show()

        
    
    def run_analysis(self):
        self.feature_engineering()
        self.ordinal_encode()
        self.scale_numerical()
        self.one_hot_encode()
        self.get_df_basic_info()
        self.plot_distributions()
        self.df.to_csv("data/Employee_processed.csv", index=False)
       
            


if __name__ == "__main__":
    analyzer = PreProcessing("data/Employee.csv")
    analyzer.run_analysis()