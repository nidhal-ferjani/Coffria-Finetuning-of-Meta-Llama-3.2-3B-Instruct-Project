# pipeline_fine_tuning.py
"""
Script modulaire pour l'affinage du modèle Llama 3.2 3B pour la recherch"
de documents sur la plateforme Coffria.
"""
import os
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from src.finetuning.model_loader import ModelLoader
from src.finetuning.dataset_preparation import DatasetPreparation
from src.finetuning.model_trainer import ModelTrainer
from src.finetuning.model_saver import ModelSaver
from src.finetuning.memory_management import MemoryManagement
from src.finetuning.model_evaluator import ModelEvaluator
from src.finetuning.cloud_uploader import CloudUploader
from src.finetuning.config_loader import ConfigLoader


def run_pipeline():
    """Fonction principale qui orchestre l'ensemble du processus d'affinage."""
    try:
        logging.info("Démarrage du processus d'affinage")
        dataset_path = "src/finetuning/data/dialogues_naturel_dataset_vfinale.jsonl"
        max_samples = 200
        
        # 1. Initialisation du modèle
        model_loader = ModelLoader("src/finetuning/config/model_loading_params.yaml")
        quantized_model, tokenizer = model_loader.load_model()
        lora_model = model_loader.prepare_for_lora(quantized_model, config_path="src/finetuning/config/lora_params.yaml")
        
        # 2. Préparation du jeu de données
        dataset_prep = DatasetPreparation(tokenizer)
        dataset = dataset_prep.load_dataset(dataset_path=dataset_path, max_samples=max_samples, split = "train")  # Remplacer par votre dataset
        formatted_dataset = dataset_prep.format_dataset(dataset)
      
        print(f"Exemple formaté: {formatted_dataset[0]}") # Afficher un exemple pour vérification
        logging.info(f"Exemple formaté: {formatted_dataset[0]}")        
        
        # 3. Configuration et lancement de l'affinage
        trainer_obj = ModelTrainer(lora_model, tokenizer, formatted_dataset)
        trainer = trainer_obj.setup_trainer(config_path="src/finetuning/config/trainer_params.yaml")
        training_stats = trainer_obj.train(trainer)
        
        # 4. Sauvegarde du modèle affiné
        lora_save_path = "src/finetuning/llama-store-finetuned/model_lora"
        ModelSaver.save_model(lora_model, tokenizer, lora_save_path)
        
        # 5. Nettoyage de la mémoire
        del trainer
        MemoryManagement.clean_memory()
        
        # 6. Fusion et sauvegarde du modèle final
        merged_save_path = "src/finetuning/llama-store-finetuned/model_merged"
        final_model = ModelSaver.merge_and_save_model(
            quantized_model, lora_save_path, merged_save_path, tokenizer
        )
        
        # 7. Évaluation du modèle
        evaluator = ModelEvaluator(final_model, tokenizer)
        inference_model = evaluator.prepare_for_inference()
        
        test_prompts = [
            {"content": "Pouvez-vous trouver des documents sur la réglementation de l'IA en Europe ?", "langue": "fr"},
            {"content": "Je cherche des articles académiques sur l'apprentissage automatique et la santé.", "langue": "fr"},
            {"content": "Quels sont les derniers rapports sur l'économie circulaire en France ?", "langue": "fr"},
            {"content": "Avez-vous des documents sur les tendances du marché de la cybersécurité ?", "langue": "fr"},
            {"content": "Je voudrais des études sur l'impact des cryptomonnaies sur le secteur bancaire.", "langue": "fr"},

            {"content": "Can you find research papers on the ethical implications of artificial intelligence?", "langue": "en"},
            {"content": "I'm looking for reports on climate change policies in North America.", "langue": "en"},
            {"content": "Do you have financial analysis documents for major tech companies?", "langue": "en"},
            {"content": "I need case studies on blockchain adoption in supply chains.", "langue": "en"},
            {"content": "Are there any recent white papers on quantum computing advancements?", "langue": "en"}
        ]

        
        for prompt in test_prompts:
            response = evaluator.generate_response(inference_model, prompt.get("content"), prompt.get("langue"))
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Réponse: {response[:100]}...")
        
        # 8. Nettoyage de la mémoire
        del inference_model, final_model
        MemoryManagement.clean_memory()
        
        
        config = ConfigLoader.load_config(config_path="src/finetuning/config/save_params.yaml")
        
        # 9. Téléchargement vers HuggingFace Hub (optionnel)
        params_face = config.get('hugging_face_params', {})
        repo_name = params_face.get("depot_name")
        access_token = params_face.get("access_token")
    
        ModelSaver.save_model(final_model, tokenizer, "src/finetuning/llama-store-finetuned/model_to_upload")
        CloudUploader.push_to_huggingface_hub(model=final_model, tokenizer=tokenizer, repo_name=repo_name, auth_token=access_token, message="Fine-tuning Llama 3B sur dialogues Coffria, LoRA")     
          
        # 10. Téléchargement vers Google Drive (optionnel)
        params = config.get('google_params', {})
        model_local = params.get("local_model_folder")
        archive_name = params.get("modele_name_compresse")
        google_drive_folder_id = params.get("google_drive_folder_id")
        google_credentials_path = "src/finetuning/config/client_secret_google_drive.json"  
        CloudUploader.upload_to_google_drive(local_dir=model_local,google_drive_folder_id=google_drive_folder_id , google_credentials_path=google_credentials_path, archive_name=archive_name)
        
        
        logging.info("Processus d'affinage terminé avec succès")
    
    except Exception as e:
        logging.error(f"Erreur dans le processus principal d'affinage: {e}")
        raise CustomException(f"Erreur dans le processus principal d'affinage: {e}", e)


if __name__ == "__main__":
    run_pipeline()