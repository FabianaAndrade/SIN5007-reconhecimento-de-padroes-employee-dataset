import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dados dos modelos
data = {
    'Naive Bayes': {
        'Full Dataset': {'Acurácia': 70.89, 'Precisão': 59.19, 'Recall': 50.16, 'F1-Score': 54.3},
        'PCA': {'Acurácia': 71.86, 'Precisão': 63.98, 'Recall': 42.06, 'F1-Score': 50.75},
        'SelectedKBest': {'Acurácia': 70.46, 'Precisão': 58.21, 'Recall': 50.78, 'F1-Score': 54.24}
    },
    'Random Forest': {
        'Full Dataset': {'Acurácia': 83.13, 'Precisão': 80.27, 'Recall': 67.69, 'F1-Score': 73.40},
        'PCA': {'Acurácia': 82.66, 'Precisão': 79.60, 'Recall': 66.75, 'F1-Score': 72.56},
        'SelectedKBest': {'Acurácia': 83.86, 'Precisão': 84.29, 'Recall': 65.25, 'F1-Score': 73.56}
    },

    'SVM': {
        'Full Dataset': {'Acurácia': 0, 'Precisão': 0, 'Recall': 0, 'F1-Score': 0},
        'PCA': {'Acurácia': 0, 'Precisão': 0, 'Recall': 0, 'F1-Score': 0},
        'SelectedKBest': {'Acurácia': 0, 'Precisão': 0, 'Recall': 0, 'F1-Score': 0}
    }
}

# Datasets
datasets = ['Full Dataset', 'PCA', 'SelectedKBest']
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
models = ['Naive Bayes', 'Random Forest', 'SVM']

colors = {
    'Naive Bayes': '#1f77b4',      
    'Random Forest': '#ff7f0e',
    'SVM': '#2ca02c'
}
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

bar_width = 0.35
x = np.arange(len(datasets))

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    nb_values = [data['Naive Bayes'][ds][metric] for ds in datasets]
    rf_values = [data['Random Forest'][ds][metric] for ds in datasets]

    svm_values = [data['SVM'][ds][metric] for ds in datasets]

    bars1 = ax.bar(x - bar_width/2, nb_values, bar_width, label='Naive Bayes', 
                   color=colors['Naive Bayes'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + bar_width/2, rf_values, bar_width, label='Random Forest', 
                   color=colors['Random Forest'], alpha=0.8, edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x, svm_values, bar_width, label='SVM', 
                   color=colors['SVM'], alpha=0.8, edgecolor='black', linewidth=1.2)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Valor (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

plt.suptitle('Comparação de Modelos', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('figures/comparacao_modelos_metricas.png', dpi=300, bbox_inches='tight')
plt.show()

