import json
import logging
import os
from typing import List, Dict, Tuple
from src.finetuning.logger import logging

def validate_dialogue_json(dataset_path: str) -> Tuple[int, List[Dict]]:
    """
    Valide la structure d'un fichier JSON de dialogues et compte les dialogues invalides.

    Args:
        dataset_path (str): Chemin vers le fichier JSON du dataset.

    Returns:
        Tuple[int, List[Dict]]: Un tuple contenant :
            - Le nombre de dialogues invalides.
            - Une liste d'exemples de dialogues invalides (pour inspection).
    """
    invalid_dialogue_count = 0
    invalid_dialogues_examples = []

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)

        if not isinstance(dialogues, list):
            raise ValueError("Le fichier JSON doit contenir une liste à la racine.")

        for index, dialogue in enumerate(dialogues):
           
            if not isinstance(dialogue, dict):
                invalid_dialogue_count += 1
                invalid_dialogues_examples.append({"index": index, "error": "Le dialogue n'est pas un dictionnaire.", "dialogue": dialogue})
                continue # Passer à la vérification du dialogue suivant

            if 'conversations' not in dialogue:
                invalid_dialogue_count += 1
                invalid_dialogues_examples.append({"index": index, "error": "Clé 'conversations' manquante.", "dialogue": dialogue})
                continue

            conversations = dialogue['conversations']
            if not isinstance(conversations, list):
                invalid_dialogue_count += 1
                invalid_dialogues_examples.append({"index": index, "error": "La valeur de 'conversations' n'est pas une liste.", "dialogue": dialogue})
                continue

            for turn_index, turn in enumerate(conversations):
                if not isinstance(turn, dict):
                    invalid_dialogue_count += 1
                    invalid_dialogues_examples.append({"index": index, "turn_index": turn_index, "error": "Le tour de dialogue n'est pas un dictionnaire.", "dialogue": dialogue, "turn": turn})
                    #break # Pas la peine de vérifier les autres tours de ce dialogue si un tour est déjà invalide

                if not all(key in turn for key in ['role', 'content']):
                    invalid_dialogue_count += 1
                    invalid_dialogues_examples.append({"index": index, "turn_index": turn_index, "error": "Clé 'role' ou 'content' manquante dans le tour de dialogue.", "dialogue": dialogue, "turn": turn})
                    #break # Pas la peine de vérifier les autres tours de ce dialogue si un tour est déjà invalide

    except json.JSONDecodeError as e:
     
        logging.error(f"Erreur de décodage JSON dans le fichier {dataset_path}: {e}")
        return -1, [{"error": f"Erreur de décodage JSON: {e}"}] # -1 pour indiquer une erreur fatale de JSON

    except FileNotFoundError:
        logging.error(f"Fichier non trouvé: {dataset_path}")
        return -1, [{"error": f"Fichier non trouvé: {dataset_path}"}] # -1 pour indiquer une erreur fatale de fichier

    except ValueError as e: # Capturer l'erreur si le JSON n'est pas une liste à la racine
        logging.error(f"Erreur de structure du fichier JSON: {e}")
        return -1, [{"error": f"Erreur de structure JSON: {e}"}] # -1 pour indiquer une erreur fatale de structure JSON


    logging.info(f"Validation terminée. Nombre de dialogues invalides: {invalid_dialogue_count}")
    logging.error(invalid_dialogues_examples)
    return invalid_dialogue_count, invalid_dialogues_examples

def main():
    dataset_path = "src/finetuning/data/dialogues_naturel_dataset_vfinal.json" # **REMPLACEZ CECI PAR LE CHEMIN VERS VOTRE FICHIER JSON**

    invalid_count, invalid_examples = validate_dialogue_json(dataset_path)
    logging.error(invalid_examples)

    if invalid_count == -1:
        print(f"Erreur fatale lors de la validation du fichier JSON. Voir les logs pour plus de détails.")
    elif invalid_count > 0:
        print(f"Nombre de dialogues invalides détectés: {invalid_count}")
        print("\nExemples de dialogues invalides (les 5 premiers) :")
        for example in invalid_examples[:5]: # Afficher les 5 premiers exemples invalides
            print(f"\nDialogue Index: {example['index']}")
            if 'turn_index' in example:
                 print(f"Turn Index: {example['turn_index']}") # Afficher l'index du tour si disponible
            print(f"Error: {example['error']}")
            print(f"Dialogue Content (extrait):\n{str(example['dialogue'])[:500]}...\n{'-'*50}") # Afficher un extrait du dialogue
    else:
        print("Fichier JSON. Aucun dialogue invalide détecté.")


if __name__ == "__main__":
    main()