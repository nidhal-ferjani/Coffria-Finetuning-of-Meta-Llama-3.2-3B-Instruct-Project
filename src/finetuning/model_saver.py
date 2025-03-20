import os
from typing import Any
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from peft import PeftModel

class ModelSaver:
    """Classe pour sauvegarder et exporter le modèle affiné."""
    
    @staticmethod
    def save_model(model: Any, tokenizer: Any, save_path: str) -> None:
        """
        Sauvegarde le modèle et le tokenizer.
        
        Args:
            model: Le modèle affiné
            tokenizer: Le tokenizer du modèle
            save_path: Chemin où sauvegarder le modèle
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            logging.info(f"Sauvegarde du modèle affiné dans {save_path}")
            
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            logging.info(f"Modèle et tokenizer sauvegardés avec succès dans {save_path}")
        
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            raise CustomException(f"Erreur lors de la sauvegarde du modèle: {e}", e)
    
    @staticmethod
    def merge_and_save_model(base_model: Any, finetuned_model_path: str, save_path: str, tokenizer: Any) -> Any:
        """
        Fusionne les poids LoRA avec le modèle de base et sauvegarde le résultat.
        
        Args:
            base_model: Le modèle de base
            finetuned_model_path: Chemin vers le modèle affiné (poids LoRA)
            save_path: Chemin où sauvegarder le modèle fusionné
            tokenizer: Le tokenizer du modèle
            
        Returns:
            Le modèle fusionné
        """
        try:
            logging.info(f"Fusion des poids LoRA avec le modèle de base")
            
            final_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
            final_model = final_model.merge_and_unload()
            
            os.makedirs(save_path, exist_ok=True)
            logging.info(f"Sauvegarde du modèle fusionné dans {save_path}")
            
            final_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            logging.info(f"Modèle fusionné et tokenizer sauvegardés avec succès dans {save_path}")
            return final_model
        
        except Exception as e:
            logging.error(f"Erreur lors de la fusion et de la sauvegarde du modèle: {e}")
            raise CustomException(f"Erreur lors de la fusion et de la sauvegarde du modèle: {e}", e)
