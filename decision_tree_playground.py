import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing, tree

class DecisionTreeExperiment:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(csv_path)

        # Para treinar o modelo, precisamos de dados e um alvo
        self.target = self.dataframe["LeaveOrNot"].copy()
        self.target = self.target.to_frame()
        self.target_categories = None
        self.target_encoded = None
        
        self.data = self.dataframe.copy()
        self.data = self.data.drop(columns=["LeaveOrNot"])
        self.data_categories = None
        self.data_encoded = None
        
        self.classifier = None
        
    def ordinal_encode(self, dataframe):
        #print(dataframe)
        ordinal_encoder = preprocessing.OrdinalEncoder()
        ordinal_encoder.fit(dataframe)
        encoded_dataframe = ordinal_encoder.transform(dataframe)
        #print(encoded_dataframe)
        return encoded_dataframe, ordinal_encoder.categories_

    def run(self):
        data_encoded, data_categories = experiment.ordinal_encode(experiment.data)
        target_encoded, target_categories = experiment.ordinal_encode(experiment.target)
        # Qualquer coisa maior que 4 começa a ficar difícil de visualizar
        classifier = tree.DecisionTreeClassifier(max_depth=4)
        classifier.fit(data_encoded, target_encoded)
        self.classifier = classifier
        self.target_encoded, self.target_categories = target_encoded, target_categories
        self.data_encoded, self.data_categories = data_encoded, data_categories

    def plot(self):

        # Peço desculpas, mas vou ver o matplotlib no futuro
        # Então o Gemini fez esse cara aqui e eu consertei algumas coisas
        
        feature_names = self.data.columns.to_list()
    
        # --- Plotting and Exporting ---
    
        # 1. Create a large figure to hold both the tree and the text
        fig, ax = plt.subplots(figsize=(24, 12))
    
        # 2. Plot the tree
        tree.plot_tree(
            self.classifier, 
            feature_names=feature_names,
            class_names=[str(c) for c in self.target_categories[0]], 
            filled=True, 
            ax=ax,
            fontsize=10
        )
    
        # 3. Construct the encoding key text
        mapping_text = "Legenda de Codificação:\n\n"
        for i, feature in enumerate(feature_names):
            categories = self.data_categories[i]
            # Format: 0=CategoryA, 1=CategoryB
            mapping = ", ".join([f"{val}={cat}" for val, cat in enumerate(categories)])
        
            # Add a newline every 50 characters to prevent the text from trailing off screen
            import textwrap
            mapping = textwrap.fill(mapping, width=60)
        
            mapping_text += f"• {feature}:\n{mapping}\n\n"
        
        # 4. Attach the text box to the figure (left side)
        plt.figtext(
            0.02, 0.5, mapping_text, 
            fontsize=10, 
            verticalalignment='center', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgrey", alpha=0.5)
        )
    
        # 5. Adjust layout to make room for the text block and save
        plt.subplots_adjust(left=0.25) 
        plt.savefig("figures/decision_tree_with_key.png", dpi=300, bbox_inches='tight')

        # Fim do código do Gemini
        
if __name__ == "__main__":
    experiment = DecisionTreeExperiment(\
            "data/Employee.csv"\
                                        )
    experiment.run()
    experiment.plot()
