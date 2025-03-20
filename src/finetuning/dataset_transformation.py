import os
import json
from src.finetuning.logger import logging
from typing import Any, List, Dict, Optional
from src.finetuning.exception import CustomException
from datasets import Dataset, load_dataset



class DatasetTransformation:
    """Classe pour préparer le jeu de données pour l'affinage, avec sauvegarde JSON."""

    def __init__(self, tokenizer: Any=None):
        """
        Initialise la préparation des données.

        Args:
            tokenizer: Le tokenizer du modèle
        """
        
        logging.info("DatasetPreparation initialisé")

    def _transform_dialogue_format(self, example: Dict) -> List[Dict]:
        """
        Transforme un dialogue de votre format JSON en format Hugging Face Datasets.

        Args:
            example (Dict): Un exemple de dialogue dans votre format JSON.

        Returns:
            List[Dict]: Liste de dictionnaires, chaque dictionnaire représentant un tour de dialogue formaté.
        """
        #print(f"Type de example: {type(example)}") # Ajouter cette ligne
        #print(f"Contenu de example: {example}") # Ajouter cette ligne
        transformed_turns = []
        for turn in example:
            transformed_turns.append({
                "content": turn['content'],
                "role": turn['role']
            })
        return transformed_turns

    def save_dataset_to_json(self, dataset: Dataset, save_path: str):
        """
        Enregistre un objet Dataset Hugging Face dans un fichier JSON.

        Args:
            dataset (Dataset): Le jeu de données à sauvegarder.
            save_path (str): Chemin complet vers le fichier JSON de sauvegarde.
        """
        try:
            logging.info(f"Sauvegarde du dataset transformé au format JSON dans : {save_path}")
            #dataset.to_json(save_path, orient='records', lines=True) # **Ajout ESSENTIEL : encoding='utf-8'**
            
            # Convertir le dataset en liste de dictionnaires
            data = [example for example in dataset]

            # Sauvegarder le dataset transformé au format JSON avec encodage UTF-8
            with open(save_path, 'w', encoding='utf-8') as file:
                for example in data:
                    file.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            logging.info(f"Dataset sauvegardé avec succès dans : {save_path}")

        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde du dataset au format JSON: {e}")
            raise CustomException(f"Erreur lors de la sauvegarde du dataset JSON: {e}", e)


    def load_dataset(self, dataset_path: str, max_samples: int, split: str = "train", save_transformed_path: Optional[str] = None) -> Dataset:
        """
        Charge, transforme, tronque et sauvegarde optionnellement le jeu de données.

        Args:
            dataset_path: Chemin vers le fichier JSON du jeu de données.
            max_samples (int): Nombre maximal d'échantillons à charger.
            split: La partition du jeu de données.
            save_transformed_path (Optional[str]): Chemin optionnel pour sauvegarder le dataset transformé en JSON.

        Returns:
            Dataset: Le jeu de données chargé, transformé, et potentiellement tronqué.
        """
        try:
            logging.info(f"Chargement du jeu de données depuis {dataset_path}")

            # Charger le dataset depuis un fichier JSON local
            dataset = load_dataset('json', data_files=dataset_path, split=split)

            logging.info(f"Jeu de données chargé, {len(dataset)} dialogues trouvés avant transformation et troncation.")

            # Transformation du format de dialogue
            dataset = dataset.map(
                lambda examples: {'conversations': [self._transform_dialogue_format(example) for example in examples['conversations']]},
                batched=True,
                num_proc=os.cpu_count() # Utiliser plusieurs cœurs CPU pour accélérer
            )
            logging.info("Format du jeu de données transformé en style Hugging Face Datasets.")

            # Tronquer le dataset si max_samples est spécifié (logique inchangée)
            # Tronquer le dataset si max_samples est spécifié (logique inchangée)
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    logging.info(f"Jeu de données tronqué à {max_samples} dialogues (max_samples > 0).")
                else:
                    logging.info(f"Dataset size ({len(dataset)}) <= max_samples ({max_samples}), aucune troncation nécessaire pour max_samples > 0.")
            elif max_samples <= 0:
                logging.info(f"max_samples <= 0 ({max_samples}), chargement de TOUS les dialogues du dataset.")
                pass
            else:
                logging.warning(f"Valeur inattendue pour max_samples: {max_samples}. Dataset complet utilisé par défaut.")
           
            # Sauvegarde optionnelle du dataset transformé en JSON
            if save_transformed_path:
                self.save_dataset_to_json(dataset, save_transformed_path)


            logging.info(f"Jeu de données final après chargement, transformation et troncation : {len(dataset)} dialogues.")
            return dataset

        except Exception as e:
            logging.error(f"Erreur lors du chargement et de la transformation du jeu de données: {e}")
            raise CustomException(f"Erreur lors du chargement et de la transformation du jeu de données: {e}", e)


def main():
    # Exemple d'utilisation (à adapter avec votre tokenizer et dataset)


    # Initialiser DatasetPreparation
    data_prep = DatasetTransformation(tokenizer=None) # Passer votre tokenizer ici

    # Chemin vers votre dataset (remplacez par votre chemin)
    dataset_path = "src/finetuning/data/dialogues_naturel_dataset_vfinal.json" # **Remplacez par le chemin de VOTRE fichier JSON**
    save_transformed_path = "src/finetuning/data/dialogues_naturel_dataset_vfinale.jsonl" # **Chemin de sauvegarde pour le dataset transformé**

    # Nombre maximal de dialogues à charger (par exemple, 2 ou -1 pour tout charger)
    max_samples = -1

    # Charger et transformer le dataset, et le sauvegarder en JSON
    train_dataset = data_prep.load_dataset(
        dataset_path=dataset_path,
        max_samples=max_samples,
        split="train",
        save_transformed_path=save_transformed_path # Passer le chemin de sauvegarde ici
    )

    print(f"Taille du dataset chargé : {len(train_dataset)}")
    print("Premier exemple du dataset transformé :")
    print(train_dataset[0])
    print("\nDataset complet transformé (les premiers exemples) :")
    for example in train_dataset:
        print(example)
        print("-" * 50) # Séparateur pour la lisibilité
        if list(train_dataset).index(example) > 3 : break # Afficher uniquement les 4 premiers pour l'exemple

def is_dialogue_valid(example): # Fonction de filtrage (à adapter!)
    # **Remplacez cette condition par un test réel pour détecter les exemples erronés**
    # Exemple hypothétique: filtrer si 'conversations' n'est pas une liste
    return isinstance(example['conversations'], list) and len(example['conversations']) > 0

def load_and_inspect_dataset(jsonl_path: str):
    """
    Charge un fichier JSONL dans un Dataset Hugging Face, vérifie son contenu,
    et affiche quelques exemples sur la console.

    Args:
        jsonl_path (str): Chemin vers le fichier JSONL.
    """
    try:
        # Charger le fichier JSONL dans un Dataset
        print(f"Chargement du fichier JSONL depuis : {jsonl_path}")
        dataset = load_dataset('json', data_files=jsonl_path, split='train')
        
        # Vérifier que le dataset a été chargé correctement
        if dataset is None or len(dataset) == 0:
            raise ValueError("Le dataset est vide ou n'a pas été chargé correctement.")
        
        print(f"✅ Dataset chargé avec succès. Nombre d'exemples : {len(dataset)}")
        
        # Afficher quelques exemples du dataset
        print("\nAffichage des premiers exemples du dataset :")
        for i, example in enumerate(dataset):
            print(f"\n--- Exemple {i + 1} ---")
            print(example)
            
            # Arrêter après 3 exemples pour ne pas surcharger la console
            if i >= 2:
                break
    
    except Exception as e:
        print(f"Erreur lors du chargement ou de l'affichage du dataset : {e}")


# ... reste du code ...
if __name__ == "__main__":
    #main()

    """dataset = load_dataset('json', data_files="src/finetuning/data/dialogues_naturel_dataset_vfinal.json", split="train")
    dataset = dataset.filter(is_dialogue_valid, num_proc=os.cpu_count()) # Filtrer le dataset
    print(f"Dataset filtré, {len(dataset)} exemples valides restants.")"""
     
    # Chemin vers votre fichier JSONL
    jsonl_path = "src/finetuning/data/dialogues_naturel_dataset_vfinale.jsonl"
    
    # Charger et inspecter le dataset
    load_and_inspect_dataset(jsonl_path)  