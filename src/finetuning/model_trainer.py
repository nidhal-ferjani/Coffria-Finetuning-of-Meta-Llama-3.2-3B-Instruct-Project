import os
from src.finetuning.logger import logging
from typing import Any
from src.finetuning.exception import CustomException
from datasets import Dataset
from src.finetuning.config_loader import ConfigLoader 
from src.finetuning.guaranteed_loss_logger import GuaranteedLossLogger
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
from unsloth import (
    is_bfloat16_supported,
    train_on_responses_only
)

class ModelTrainer:
    """Classe pour gérer l'affinage du modèle."""
    
    def __init__(self, model: Any, tokenizer: Any, dataset: Dataset):
        """
        Initialise l'entraîneur de modèle.
        
        Args:
            model: Le modèle préparé pour l'affinage
            tokenizer: Le tokenizer du modèle
            dataset: Le jeu de données formaté
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        logging.info("ModelTrainer initialisé")
    
    def setup_trainer(self, config_path: str="trainer_params.yaml", log_trainer_path: str="src/finetuning/config/training_logs.json") -> SFTTrainer:
        """
        Configure le trainer pour l'affinage.
        
        Returns:
            Le trainer configuré
        """
        try:
            log_callback = GuaranteedLossLogger(log_file=log_trainer_path)
            trainer_config = ConfigLoader.load_config(config_path=config_path)
            params = trainer_config.get('traning_params', {})
            
            max_seq_length = params.get("max_seq_length", 2048)
            per_device_train_batch_size = params.get("per_device_train_batch_size", 12)
            gradient_accumulation_steps = params.get("gradient_accumulation_steps", 4)
            #warmup_steps = params.get("warmup_steps", 50)
            warmup_ratio = params.get("warmup_ratio", 0.1)
            #max_steps = params.get("max_steps", 60)
            learning_rate = params.get("learning_rate", 2e-5)
            logging_steps = params.get("logging_steps", 100)
            optim = params.get("optim", "adamw_8bit")
            weight_decay = params.get("weight_decay", 0.01)
            lr_scheduler_type = params.get("lr_scheduler_type", "linear")
            seed = params.get("seed", 3407)
            output_dir = params.get("output_dir", "outputs")
            num_epochs = params.get("num_epochs", 2)
            report_to = params.get("report_to", "none")
            dataset_num_proc = params.get("dataset_num_proc", 2)
            save_strategy = params.get("save_strategy", "epoch")
            logging_steps = params.get("logging_steps", 100)
            
            logging.info("Configuration du trainer avec les paramètres chargés")
            
            # Création du dossier de sortie s'il n'existe pas
            os.makedirs(output_dir, exist_ok=True)
            
            # Configuration du trainer
            training_args = TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                #warmup_steps=warmup_steps,
                warmup_ratio=warmup_ratio,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=logging_steps,
                optim=optim,
                save_strategy=save_strategy,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                num_train_epochs=num_epochs,
                seed=seed,
                output_dir=output_dir,
                report_to=report_to,
                logging_steps=logging_steps,
            )
            
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
                callbacks=[log_callback],  # Ajouter le callback ici,
                compute_metrics=lambda eval_preds: {"loss": eval_preds.loss},  
                dataset_num_proc=dataset_num_proc,
                packing=False,
                args=training_args,
            )
            
            # Configuration pour entraîner uniquement sur les réponses
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
            )
            

            logging.info("Trainer configuré avec succès")
            return trainer
        
        except Exception as e:
            logging.error(f"Erreur lors de la configuration du trainer: {e}")
            raise CustomException(f"Erreur lors de la configuration du trainer: {e}", e)
    
    def train(self, trainer: SFTTrainer) -> Any:
        """
        Lance l'affinage du modèle.
        
        Args:
            trainer: Le trainer configuré
            
        Returns:
            Les statistiques d'entraînement
        """
        try:
            logging.info("Démarrage de l'affinage du modèle")
            training_stats = trainer.train()
            logging.info("Affinage terminé avec succès")
            return training_stats
        
        except Exception as e:
            logging.error(f"Erreur pendant l'affinage: {e}")
            raise CustomException(f"Erreur pendant l'affinage: {e}", e)
