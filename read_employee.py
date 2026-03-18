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
        print("plots")
        nominal_columns = ['City', 'Gender', 'EverBenched']
        num_cols = len(nominal_columns)
        rows = (num_cols + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(10, 5 * rows))
        axes = axes.flatten()
        
        max_count = max(self.df[col].value_counts().max() for col in nominal_columns) + 500
        
        for i, col in enumerate(nominal_columns):
            vc = self.df[col].value_counts()
            total = vc.sum()
            labels = [f'{count}\n({count/total*100:.1f}%)' for count in vc.values]
            bars = axes[i].bar(vc.index, vc.values, color='gray')
            axes[i].bar_label(bars, labels=labels)  
            axes[i].set_title(f'Distribuição de "{col}"')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Total')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, max_count)  
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('Figura 1 - Distribuição das variáveis nominais', y=0.95)
        plt.savefig('figures/distribuicoes_nominal.png')
        plt.close()

    def generate_ordinal_plot(self):
        print("plot simples PaymentTier e Education")
        ordinal_columns = ['PaymentTier', 'Education']
        max_count = max(self.df[col].value_counts().max() for col in ordinal_columns) + 500
        fig, axes = plt.subplots(1, len(ordinal_columns), figsize=(5 * len(ordinal_columns), 5))
        if len(ordinal_columns) == 1:
            axes = [axes]
        for i, col in enumerate(ordinal_columns):
            if 'LeaveOrNot' in self.df.columns:
                cross_tab = pd.crosstab(self.df[col], self.df['LeaveOrNot'])
                cross_tab.plot(kind='bar', stacked=True, ax=axes[i], color=['lightgray', 'dimgray'])
                
                total_samples = len(self.df)
                for c in axes[i].containers:
                    #labels = [f'{int(v.get_height())}\n({v.get_height()/total_samples*100:.1f}%)' if v.get_height() > 0 else '' for v in c]
                    axes[i].bar_label(c, label_type='center', fontsize=10)
            else:
                sns.countplot(x=col, data=self.df, ax=axes[i], color='gray')
                total_samples = len(self.df)
                for c in axes[i].containers:
                    #labels = [f'{int(v.get_height())}\n({v.get_height()/total_samples*100:.1f}%)' if v.get_height() > 0 else '' for v in c]
                    axes[i].bar_label(c, label_type='center', fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Total')
            if col == 'Education':
                axes[i].tick_params(axis='x', rotation=0)
            else:
                axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, max_count)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('Figura 2 - Distribuição das variáveis ordinais', y=0.95)
        plt.savefig('figures/distribuicao_ordinal.png')
        plt.close()


    def generate_numeric_plot(self):
        print("plot numerico")
        numeric_columns = ['Age', 'ExperienceInCurrentDomain', 'JoiningYear']
        num_cols = len(numeric_columns)
        rows = (num_cols + 1) // 2
        max_count = max(self.df[col].value_counts().max() for col in numeric_columns) + 500
        fig, axes = plt.subplots(rows, 2, figsize=(15, 6 * rows + 2))
        axes = axes.flatten()

        for i, col in enumerate(numeric_columns):
            vc = self.df[col].value_counts().sort_index()
            total = vc.sum()
            #labels = [f'{count}\n({count/total*100:.1f}%)' for count in vc.values]
            bars = axes[i].bar(vc.index, vc.values, color='gray', width=0.8)
            axes[i].bar_label(bars, rotation=45, padding=3)  
            axes[i].set_title(f'Distribuição de "{col}"', pad=20)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Total')
            axes[i].set_xticks(vc.index)
            axes[i].set_xticklabels(vc.index, rotation=60)
            axes[i].set_ylim(0, max_count)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('Figura 3 - Distribuição das variáveis numéricas', y=0.99)
        plt.savefig('figures/distribuicoes_numericas.png')
        plt.close()

    def generate_bloxpots(self):
        print("boxplots")
        numeric_columns = ['Age', 'ExperienceInCurrentDomain', 'JoiningYear']
        num_cols = len(numeric_columns)
        rows = (num_cols + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(15, 6 * rows + 2))
        axes = axes.flatten()

        for i, col in enumerate(numeric_columns):
            sns.boxplot(y=col, data=self.df, ax=axes[i], color='gray')
            axes[i].set_title(f'Boxplot de "{col}"', pad=20)
            axes[i].set_ylabel(col)
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=60)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('Figura 4 - Boxplots das variáveis numéricas', y=0.99)
        plt.savefig('figures/boxplots_numericas.png')
        plt.close()
        

    def get_class_distribution(self):
        if "LeaveOrNot" in self.df.columns:
            print("\n(LeaveOrNot):")
            vc = self.df["LeaveOrNot"].value_counts()
            total = vc.sum()
            percentages = (vc / total * 100).round(1)
            cell_data = [[cls, cnt, f'{perc}%'] for cls, cnt, perc in zip(vc.index, vc.values, percentages)]
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=cell_data, colLabels=['LeaveOrNot', 'Count', 'Porcentagem (%)'], cellLoc='center', loc='center')
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
        plt.title("Figura 5 - Heatmap de correlação entre as variáveis")
        plt.tight_layout()
        plt.savefig("figures/heatmap.png")

    def run_analysis(self):
        self.get_basic_info()
        self.generate_plots()
        self.get_class_distribution()
        self.get_missing_data_info()
        self.generate_ordinal_plot()
        self.generate_correlation_heatmap()
        self.generate_numeric_plot()
        self.generate_bloxpots()


if __name__ == "__main__":
    analyzer = EmployeeDatasetAnalyzer("data/Employee.csv")
    analyzer.run_analysis()
