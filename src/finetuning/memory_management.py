import torch
import gc
from src.finetuning.logger import logging

class MemoryManagement:
    """Classe pour gérer la mémoire pendant l'affinage."""
    
    @staticmethod
    def clean_memory() -> None:
        """Nettoie la mémoire pour éviter les fuites."""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Nettoyage de la mémoire effectué")
        
        except Exception as e:
            logging.error(f"Erreur lors du nettoyage de la mémoire: {e}")

