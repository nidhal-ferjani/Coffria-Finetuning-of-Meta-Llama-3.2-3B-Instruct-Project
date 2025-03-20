import os
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from typing import Any
from datasets import load_dataset, Dataset
from unsloth.chat_templates import standardize_sharegpt
from unsloth.chat_templates import get_chat_template

class DatasetPreparation:
    """Classe pour préparer le jeu de données pour l'affinage."""
    
    def __init__(self, tokenizer: Any):
        """
        Initialise la préparation des données.
        
        Args:
            tokenizer: Le tokenizer du modèle
        """
        self.tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
        logging.info("DatasetPreparation initialisé")
    
    def load_dataset(self, dataset_path: str, max_samples: int, split: str = "train") -> Dataset:
        """
        Charge le jeu de données et extrait un nombre maximal d'échantillons.

        Args:
            dataset_path: Chemin ou identifiant du jeu de données
            max_samples (int): Nombre maximal d'échantillons à charger.
                            - Si > 0, charge au maximum `max_samples` exemples.
                            - Si <= 0, charge TOUS les exemples.
            split: La partition du jeu de données à charger

        Returns:
            Le jeu de données chargé et tronqué
        """
        try:
            logging.info(f"Chargement du jeu de données depuis {dataset_path}")

            # Déterminer si c'est un chemin local ou Hub
            if os.path.exists(dataset_path):
                dataset = load_dataset('json', data_files=dataset_path, split=split) # Pour JSON local
            else:
                dataset = load_dataset(dataset_path, split=split)

            # Standardisation du format ShareGPT
            dataset = standardize_sharegpt(dataset)
            logging.info(f"Jeu de données chargé et standardisé, {len(dataset)} exemples trouvés avant troncation")

            # Gestion explicite de max_samples
            if max_samples > 0:
                if len(dataset) > max_samples:
                    dataset = dataset.select(range(max_samples))
                    logging.info(f"Jeu de données tronqué à {max_samples} exemples (max_samples > 0).")
                else:
                    logging.info(f"Dataset size ({len(dataset)}) <= max_samples ({max_samples}), no truncation needed for max_samples > 0.")
            elif max_samples <= 0: # Cas pour charger TOUT le dataset (y compris max_samples = 0 ou négatif)
                logging.info(f"max_samples <= 0 ({max_samples}), chargement de TOUS les exemples du dataset.")
                pass # Ne rien faire pour charger tout le dataset
            else: # Cas improbable (pourrait être couvert par elif max_samples <= 0, mais par sécurité)
                logging.warning(f"Valeur inattendue pour max_samples: {max_samples}. Dataset complet utilisé par défaut.")


            logging.info(f"Jeu de données final après chargement et troncation : {len(dataset)} exemples")
            return dataset

        except Exception as e:
            logging.error(f"Erreur lors du chargement du jeu de données: {e}")
            raise CustomException(f"Erreur lors du chargement du jeu de données: {e}", e)
        
    def format_dataset(self, dataset: Dataset) -> Dataset:
        """
        Formate le jeu de données pour l'affinage.
        
        Args:
            dataset: Le jeu de données à formater
            
        Returns:
            Le jeu de données formaté
        """
        try:
            logging.info("Formatage du jeu de données pour l'affinage")
            
            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = [self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
                return {"text": texts}
            
            formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
            logging.info("Jeu de données formaté avec succès")
            
            # Affichage d'un exemple pour vérification
            for i in formatted_dataset.select(range(2)):
                print(f"Exemple formaté: {i}")
                logging.debug(f"Exemple formaté: {i}")
                break
            
            return formatted_dataset
        
        except Exception as e:
            logging.error(f"Erreur lors du formatage du jeu de données: {e}")
            raise CustomException(f"Erreur lors du formatage du jeu de données: {e}", e)
        

if __name__ == "__main__":
    # Exemple d'utilisation
    dataset_path = "src/finetuning/data/dialogues_naturel_dataset_vfinale.jsonl" # **Remplacez par le chemin de VOTRE fichier JSON**
    max_samples = -1 # Charger TOUS les dialogues
    split = "train" # Partition à charger
    dataset_prep = DatasetPreparation(tokenizer=None) # Passer votre tokenizer ici
    dataset = dataset_prep.load_dataset(dataset_path, max_samples, split)
    formatted_dataset = dataset_prep.format_dataset(dataset)
    print(f"Exemple formaté: {formatted_dataset[0]}") # Afficher un exemple pour vérification
    logging.info("Exemple formaté affiché avec succès")        