import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module

import importlib.util
spec = importlib.util.spec_from_file_location("preprocessing", "02_pre_processing.py")
preprocessing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing_module)
PreProcessing = preprocessing_module.PreProcessing


class SVCExperiment:
    def __init__(self, csv_path: str, test_size: float = 0.2, random_state: int = 42):
    
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        
        self.df_original = pd.read_csv(csv_path)
        print(f"Dataset original carregado: {self.df_original.shape}")
        
        
        self.X = self.df_original.drop('LeaveOrNot', axis=1)
        self.y = self.df_original['LeaveOrNot']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        print(f"Treino: {self.X_train.shape[0]} amostras")
        print(f"Teste: {self.X_test.shape[0]} amostras")
        
        self.train_df = pd.concat([self.X_train, self.y_train], axis=1)
        self.test_df = pd.concat([self.X_test, self.y_test], axis=1)
        
        # dfs para pré-processamento
        self.train_df.to_csv('_train_temp.csv', index=False)
        self.test_df.to_csv('_test_temp.csv', index=False)
        
        self.model = None
        self.X_train_processed = None
        self.X_test_processed = None
    
    def preprocess_data(self):
        print("Pré-processando dados de treino")
        preprocessor_train = PreProcessing('_train_temp.csv')
        preprocessor_train.feature_engineering()
        preprocessor_train.ordinal_encode()
        preprocessor_train.scale_numerical()
        preprocessor_train.one_hot_encode()
        
        print("Pré-processando dados de teste")
        preprocessor_test = PreProcessing('_test_temp.csv')
        preprocessor_test.feature_engineering()
        preprocessor_test.ordinal_encode()
        preprocessor_test.scale_numerical()
        preprocessor_test.one_hot_encode()
        
        # retirando target
        self.X_train_processed = preprocessor_train.df.drop('LeaveOrNot', axis=1)
        self.y_train_processed = preprocessor_train.df['LeaveOrNot']
        
        self.X_test_processed = preprocessor_test.df.drop('LeaveOrNot', axis=1)
        self.y_test_processed = preprocessor_test.df['LeaveOrNot']
        
        print(f"Dados após pré-processamento:")
        print(f"X_train: {self.X_train_processed.shape}")
        print(f"X_test: {self.X_test_processed.shape}")
        
        os.remove('_train_temp.csv')
        os.remove('_test_temp.csv')
    
    def train_model(self, current_kernel: str = 'rbf', random_state: int = 42):
        ## AQUI
        self.model = SVC(
            kernel=current_kernel,
            random_state=random_state,
            n_jobs=-1
        )
        ## AQUI
        print(f"Treinando SVC com o kernel {current_kernel}")
        self.model.fit(self.X_train_processed, self.y_train_processed)
        print("Modelo treinado")
    
    def evaluate_model(self):
        
        # Predições
        y_train_pred = self.model.predict(self.X_train_processed)
        y_test_pred = self.model.predict(self.X_test_processed)
        
        # Métricas de Treino
        print("\n[Conjunto de TREINO]")
        print(f"  Acurácia: {accuracy_score(self.y_train_processed, y_train_pred):.4f}")
        print(f"  Precisão: {precision_score(self.y_train_processed, y_train_pred):.4f}")
        print(f"  Recall: {recall_score(self.y_train_processed, y_train_pred):.4f}")
        print(f"  F1-Score: {f1_score(self.y_train_processed, y_train_pred):.4f}")
        
        # Métricas de Teste
        print("\n[Conjunto de TESTE]")
        print(f"  Acurácia: {accuracy_score(self.y_test_processed, y_test_pred):.4f}")
        print(f"  Precisão: {precision_score(self.y_test_processed, y_test_pred):.4f}")
        print(f"  Recall: {recall_score(self.y_test_processed, y_test_pred):.4f}")
        print(f"  F1-Score: {f1_score(self.y_test_processed, y_test_pred):.4f}")
    
    def get_feature_importance(self, top_n: int = 10):
    
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"[Top {top_n} Features Importantes]")
        for i, idx in enumerate(indices, 1):
            print(f"  {i}. {self.X_train_processed.columns[idx]}: {importances[idx]:.4f}")
    
    def train_with_stratified_cv(self, n_splits: int = 5, n_estimators: int = 500):
        
        print("TREINAMENTO COM STRATIFIED K-FOLD CV")

    
        X_combined = pd.concat([self.X_train_processed, self.X_test_processed], ignore_index=True)
        y_combined = pd.concat([self.y_train_processed, self.y_test_processed], ignore_index=True)
        
        print(f"\nDataset combinado: {X_combined.shape[0]} amostras")
        print(f"Distribuição de classes:")
        print(y_combined.value_counts())
        
        # Configurar StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        #métricas
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
        ## AQUI
        #modelo
        model = SVC(
            random_state=42,
            n_jobs=-1
        )
        
        #cross-validation
        print(f"\nExecutando {n_splits}-Fold Stratified Cross-Validation...")
        cv_results = cross_validate(
            model, X_combined, y_combined,
            cv=skf,
            scoring=scoring,
            return_train_score=True
        )
        
        #resultados
        print(f"\n[Resultados do Cross-Validation Estratificado]")
        print(f"\nAcurácia:")
        print(f"  Train: {cv_results['train_accuracy'].mean():.4f} (+/- {cv_results['train_accuracy'].std():.4f})")
        print(f"  Test:  {cv_results['test_accuracy'].mean():.4f} (+/- {cv_results['test_accuracy'].std():.4f})")
        
        print(f"\nPrecisão:")
        print(f"  Train: {cv_results['train_precision'].mean():.4f} (+/- {cv_results['train_precision'].std():.4f})")
        print(f"  Test:  {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
        
        print(f"\nRecall:")
        print(f"  Train: {cv_results['train_recall'].mean():.4f} (+/- {cv_results['train_recall'].std():.4f})")
        print(f"  Test:  {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
        
        print(f"\nF1-Score:")
        print(f"  Train: {cv_results['train_f1'].mean():.4f} (+/- {cv_results['train_f1'].std():.4f})")
        print(f"  Test:  {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
        
        # Mostrar resultados de cada fold
        print(f"\n[Detalhes por Fold]")
        for fold in range(n_splits):
            print(f"\nFold {fold + 1}:")
            print(f"  Acurácia: {cv_results['test_accuracy'][fold]:.4f}")
            print(f"  F1-Score: {cv_results['test_f1'][fold]:.4f}")
        
        return cv_results
    
    def calibrate_kernels(self):
        
      
        print("CALIBRAÇÃO DE KERNELS COM CROSS-VALIDATION")
               
        X_combined = pd.concat([self.X_train_processed, self.X_test_processed], ignore_index=True)
        y_combined = pd.concat([self.y_train_processed, self.y_test_processed], ignore_index=True)
        
        n_features = X_combined.shape[1]
        print(f"\nDataset: {X_combined.shape[0]} amostras, {n_features} features")
        
        # Lista de kernels para testar
        kernel_list = [
            'linear',
            'poly',
            'rbf',
            'sigmoid'
        ]
        
        # Remover duplicatas mantendo ordem
        kernel_list = list(dict.fromkeys(kernel_list))
        
        print(f"Testando kernels: {kernel_list}")
        
        # Configurar StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Definir múltiplas métricas
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
        
        # Testar cada configuração de max_features
        all_results = []
        
        print(f"\nExecutando Cross-Validation para cada kernel...")
        for current_kernel in kernel_list:
            print(f"  Testando kernel={current_kernel}...")
            
            model = SVC(
                kernel=current_kernel,
                random_state=42
            )
            
            # Cross-validation
            cv_results = cross_validate(
                model, X_combined, y_combined,
                cv=skf,
                scoring=scoring,
                return_train_score=True
            )
            
            # Armazenar resultados
            result_dict = {
                'kernel': current_kernel,
                'accuracy_mean': cv_results['test_accuracy'].mean(),
                'accuracy_std': cv_results['test_accuracy'].std(),
                'precision_mean': cv_results['test_precision'].mean(),
                'precision_std': cv_results['test_precision'].std(),
                'recall_mean': cv_results['test_recall'].mean(),
                'recall_std': cv_results['test_recall'].std(),
                'f1_mean': cv_results['test_f1'].mean(),
                'f1_std': cv_results['test_f1'].std(),
                'cv_results': cv_results
            }
            
            all_results.append(result_dict)
        
        # Exibir resultados resumidos
        print(f"\n[Resumo de Todos os kernels Testados]")
        results_summary = pd.DataFrame({
            'Kernel': [f"{r['kernel']}" for r in all_results],
            'Acurácia': [f"{r['accuracy_mean']:.4f}" for r in all_results],
            'Acurácia (±)': [f"{r['accuracy_std']:.4f}" for r in all_results],
            'Precisão': [f"{r['precision_mean']:.4f}" for r in all_results],
            'Precisão (±)': [f"{r['precision_std']:.4f}" for r in all_results],
            'Recall': [f"{r['recall_mean']:.4f}" for r in all_results],
            'Recall (±)': [f"{r['recall_std']:.4f}" for r in all_results],
            'F1-Score': [f"{r['f1_mean']:.4f}" for r in all_results],
            'F1-Score (±)': [f"{r['f1_std']:.4f}" for r in all_results]
        })
        print(results_summary.to_string(index=False))
        
        # Encontrar melhor configuração (por F1-Score)
        best_result = max(all_results, key=lambda x: x['f1_mean'])
        
        print(f"\n[Melhor Configuração]")
        print(f"  Kernel: {best_result['kernel']}")
        print(f"  Acurácia:  {best_result['accuracy_mean']:.4f} ± {best_result['accuracy_std']:.4f}")
        print(f"  Precisão:  {best_result['precision_mean']:.4f} ± {best_result['precision_std']:.4f}")
        print(f"  Recall:    {best_result['recall_mean']:.4f} ± {best_result['recall_std']:.4f}")
        print(f"  F1-Score:  {best_result['f1_mean']:.4f} ± {best_result['f1_std']:.4f}")
        
        return all_results
    
    def compare_datasets(self, n_splits: int = 5, n_estimators: int = 100):
        
        datasets_info = {
            'PCA': 'data/Employee_pca.csv',
            'FS': 'data/Employee_fs.csv'
        }
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
        
        comparison_results = {}
        
        for dataset_name, dataset_path in datasets_info.items():
            print(f"\n[Testando Dataset: {dataset_name}]")
            print(f"Arquivo: {dataset_path}")
            
            # Carregar dataset
            df = pd.read_csv(dataset_path)
            X = df.drop('LeaveOrNot', axis=1)
            y = df['LeaveOrNot']
            
            n_features = X.shape[1]
            print(f"Shape: {X.shape[0]} amostras, {n_features} features")
            
            # Definir kernels a testar
            kernel_list = [
                'linear',
                'poly',
                'rbf',
                'sigmoid'
            ]
            kernel_list = list(dict.fromkeys(kernel_list))
            
            print(f"Testando kernels: {kernel_list}")
            
    
            dataset_results = []
            for current_kernel in kernel_list:
                model = SVC(
                    kernel=current_kernel,
                    random_state=42
                )
                
                cv_results = cross_validate(
                    model, X, y,
                    cv=skf,
                    scoring=scoring,
                    return_train_score=False
                )
                
                result = {
                    'kernel': current_kernel,
                    'accuracy': cv_results['test_accuracy'].mean(),
                    'precision': cv_results['test_precision'].mean(),
                    'recall': cv_results['test_recall'].mean(),
                    'f1': cv_results['test_f1'].mean()
                }
                dataset_results.append(result)
            
           
            dataset_results_sorted = sorted(dataset_results, key=lambda x: x['f1'], reverse=True)
            comparison_results[dataset_name] = dataset_results_sorted
        
        print("Melhor F1-Score para cada Dataset")
        comparison_table = []
        for dataset_name, results_sorted in comparison_results.items():
            best = results_sorted[0]
            comparison_table.append({
                'Dataset': dataset_name,
                'Kernel': str(best['kernel']),
                'Acurácia': f"{best['accuracy']:.4f}",
                'Precisão': f"{best['precision']:.4f}",
                'Recall': f"{best['recall']:.4f}",
                'F1-Score': f"{best['f1']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_table)
        print(comparison_df.to_string(index=False))
        
        return comparison_results


if __name__ == "__main__":
    experiment = SVCExperiment(
        csv_path="data/Employee.csv",
        test_size=0.2,
        random_state=42
    )
    

    experiment.preprocess_data()

    """ # Pipe 1: Treinamento e Avaliação Padrão
        experiment.train_model(n_estimators=500)
        experiment.evaluate_model()
        experiment.get_feature_importance(top_n=10)
        
        # Pipe 2: Cross-Validation Estratificado
        print("\n\nPipe 2: Cross-Validation Estratificado")
        experiment.train_with_stratified_cv(n_splits=5, n_estimators=500)
        
        # Pipe 3: Calibração de max_features
        print("\n\nPipe 3: Calibração de Hiperparâmetros")
        experiment.calibrate_max_features(n_estimators=100)
    """

    # Pipe 3: Calibração de max_features
    print("\n\nPipe 3: Calibração de Hiperparâmetros")
    experiment.calibrate_kernels()
    
    # Pipe 4: Comparação de Datasets (, PCA, FS)
    print("\n\nPipe 4: Comparação de Datasets")
    experiment.compare_datasets(n_splits=5, n_estimators=500)
