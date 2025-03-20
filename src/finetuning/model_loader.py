import torch
from typing import Any, Tuple
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.config_loader import ConfigLoader  # Assuming ConfigLoader is defined in config_loader.py
from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported
)


class ModelLoader:
    """Classe pour charger et initialiser le modèle Llama 3.2."""
    
    def __init__(self, config_path: str="model_loading_params.yaml"):
        """
        Initialise le chargeur de modèle.
        
        Args:
            config_path: Chemin vers le fichier de configuration du modèle
        """
        config = ConfigLoader.load_config(config_path)
        self.params = config.get('model_loading_params', {})
        logging.info("ModelLoader initialisé avec la configuration")
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Charge le modèle et le tokenizer en appliquant la quantification.
        
        Returns:
            Tuple contenant le modèle quantifié et le tokenizer
        """
        try:
            # Ensure that required params are present
            if not all(key in self.params for key in ['model_name', 'max_seq_length', 'dtype', 'load_in_4bit']):
                raise ValueError("Missing one or more required parameters in model_loading_params.")
            
            model_name = self.params.get("model_name", "unsloth/Llama-3.2-3B-Instruct")
            max_seq_length = self.params.get("max_seq_length", 2048)
            dtype = self.params.get("dtype", None)  # Auto détection
            load_in_4bit = self.params.get("load_in_4bit", True)
            
            logging.info(f"Chargement du modèle {model_name} avec quantification 4-bit: {load_in_4bit}")
            logging.info(f"Type de données du modèle (dtype): {torch.bfloat16 if is_bfloat16_supported() else torch.float16}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
                load_in_4bit=load_in_4bit,
            )
            
            logging.info("Modèle et tokenizer chargés avec succès")
            return model, tokenizer
        
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {e}")
            raise CustomException(f"Erreur lors du chargement du modèle: {e}", e)
    
    def prepare_for_lora(self, model: Any, config_path: str="lora_params.yaml") -> Any:
        """
        Prépare le modèle pour l'affinage LoRA.
        
        Args:
            model: Le modèle quantifié
            
        Returns:
            Le modèle préparé pour l'affinage LoRA
        """
        try:
            lora_config = ConfigLoader.load_config(config_path=config_path)
            params = lora_config.get('lora_params', {})
            r = params.get("r", 16)
            target_modules = params.get("target_modules", 
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            lora_alpha = params.get("lora_alpha", 16)
            lora_dropout = params.get("lora_dropout", 0)
            bias = params.get("bias", "none")
            use_gradient_checkpointing = params.get("use_gradient_checkpointing", "unsloth")
            random_state = params.get("random_state", 3407)
            use_rslora = params.get("use_rslora", False)
            loftq_config = params.get("loftq_config", None)
            
            logging.info("Configuration LoRA chargée, préparation du modèle pour l'affinage")
            
            lora_layers_and_quantized_model  = FastLanguageModel.get_peft_model(
                model,
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=random_state,
                use_rslora=use_rslora,
                loftq_config=loftq_config
            )
            
            logging.info("Modèle préparé pour l'affinage LoRA")
            return lora_layers_and_quantized_model
        
        except Exception as e:
            logging.error(f"Erreur lors de la préparation du modèle pour LoRA: {e}")
            raise CustomException(f"Erreur lors de la préparation du modèle pour LoRA: {e}", e)