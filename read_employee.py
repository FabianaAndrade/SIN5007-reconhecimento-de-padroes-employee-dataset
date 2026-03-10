import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


class EmployeeDatasetAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        os.makedirs('figures', exist_ok=True)
        print(f"Dataset: {csv_path}")

    def get_basic_info(self):
        print("Dataset shape:", self.df.shape)
        print(self.df.count())
        print(self.df.nunique())
        
        summary_data = []
        for col in self.df.columns:
            non_null = self.df[col].count()
            unique = self.df[col].nunique()
            summary_data.append([col, non_null, unique])
        
        fig, ax = plt.subplots(figsize=(8, len(self.df.columns)*0.4 + 0.5))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=summary_data, colLabels=['Coluna', 'Non-null Count', 'Valores únicos'], cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)
        plt.title('Tabela 1 - Informações básicas do dataset', y=0.95)
        plt.subplots_adjust(top=0.9)
        plt.savefig('figures/tb_basic_info.png', bbox_inches='tight')
        plt.close()

    def generate_plots(self):
        print("\plots")
        num_cols = len(self.df.columns)
        rows = (num_cols + 2) // 3 
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for i, col in enumerate(self.df.columns):
            vc = self.df[col].value_counts()
            vc.plot(kind='bar', color='gray', ax=axes[i])
            axes[i].set_title(f'Distribuição de "{col}"')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Total')
            axes[i].tick_params(axis='x', rotation=45)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('Figura 1 - Distribuição das variáveis', y=0.95)
        plt.savefig('figures/distribuicoes.png')
        plt.close()

    def get_class_distribution(self):
        if "LeaveOrNot" in self.df.columns:
            print("\n(LeaveOrNot):")
            vc = self.df["LeaveOrNot"].value_counts()
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=vc.reset_index().values, colLabels=['LeaveOrNot', 'Count'], cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            plt.title('Tabela 2 - Distribuição de LeaveOrNot')
            plt.savefig('figures/target_distribution.png', bbox_inches='tight')
            plt.close()

    def get_missing_data_info(self):
        missing = self.df.isna().any(axis=1).sum()
        print(f"missing values: {missing}")
        print(f"no missing values: {self.df.dropna().shape[0]}")

    def generate_correlation_heatmap(self):
        print("heatmap")
        df_encoded = pd.get_dummies(self.df, drop_first=True)
        corr = df_encoded.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Figura 2 - Heatmap de correlação entre as variáveis")
        plt.tight_layout()
        plt.savefig("figures/heatmap.png")

    def run_analysis(self):
        self.get_basic_info()
        self.generate_plots()
        self.get_class_distribution()
        self.get_missing_data_info()
        self.generate_correlation_heatmap()


if __name__ == "__main__":
    analyzer = EmployeeDatasetAnalyzer("data/Employee.csv")
    analyzer.run_analysis()
