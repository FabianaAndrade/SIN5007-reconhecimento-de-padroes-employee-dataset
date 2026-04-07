import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Education,PaymentTier,Age,ExperienceInCurrentDomain,LeaveOrNot,YearsAtCompany,City_New Delhi,City_Pune,Gender_Male,EverBenched_Yes

class PCAAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

    def decide_n_components(self):
        """
            Determina o número ideal de componentes principais para PCA 
            usando a regra de Kaiser e o gráfico de cotovelo.
        """ 
        X = self.df.drop(columns=['LeaveOrNot'])
        pca = PCA().fit(X)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Variância Explicada')
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', label='Variância Acumulada')
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variância Explicada')
        plt.title('Scree Plot - Variância Explicada por Componente')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Variância Explicada')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.legend()
        plt.grid()
        plt.savefig('figures/pca_scree_plot.png')
        plt.close()

        #maior que 90% de variância explicada
        n_components_kaiser = (cumulative_variance >= 0.9).argmax() + 1
        
        return n_components_kaiser
    
    def pca_decomposition(self, n_components):
        X = self.df.drop(columns=['LeaveOrNot'])
        y = self.df['LeaveOrNot']

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        print("PCA Components shape:", X_pca.shape)
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        print("Autovetores:", pca.components_)
        print("Autovalores:", pca.explained_variance_)
        print("Total variance explained:", sum(pca.explained_variance_ratio_))
        
        #de-para variáveis originais e componentes principais
        for i in range(n_components):
            print(f"Componente Principal {i+1}:")
            for j, feature in enumerate(X.columns):
                print(f"  {feature}: {pca.components_[i][j]:.4f}")

        colunas = [f'pca_{i+1}' for i in range(n_components)]
        

        new_df = pd.DataFrame(data=X_pca, columns=colunas)
        new_df['LeaveOrNot'] = y
        sns.pairplot(new_df, vars=colunas, hue='LeaveOrNot', diag_kind='hist')
        plt.title('PCA of Employee Data')
        plt.savefig('figures/pca_spairplot.png')
        
        #salvando df para treinamento futuro
        new_df.to_csv('data/Employee_pca.csv', index=False)

    
    def run_analysis(self):
        n_components = self.decide_n_components()
        print(f"Decided number of components for PCA: {n_components}")
        self.pca_decomposition(n_components=n_components)

if __name__ == "__main__":
    analyzer = PCAAnalyzer("data/Employee_processed.csv")
    analyzer.run_analysis()


"""Componente 1: YearsAtCompany
Componente 2: ExperienceInCurrentDomain
Componente 3: PaymentTier
Componente 4: Education"""