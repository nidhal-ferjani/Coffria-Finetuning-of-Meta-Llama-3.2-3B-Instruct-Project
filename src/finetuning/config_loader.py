from typing import Any, Dict
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
import yaml


class ConfigLoader:
    """Classe pour charger et gérer les configurations depuis les fichiers YAML."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Charge un fichier de configuration YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logging.info(f"Configuration chargée depuis {config_path}")
                return config
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration {config_path}: {e}")
            raise CustomException(f"Erreur lors du chargement de la configuration {config_path}: {e}", e)



if __name__ == "__main__":
    config = ConfigLoader.load_config("src/finetuning/config/lora_params.yaml")
    print(config)