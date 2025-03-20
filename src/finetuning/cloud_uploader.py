import os
import tarfile
from huggingface_hub import login
from src.finetuning.logger import logging
from typing import Any
from src.finetuning.exception import CustomException
from src.finetuning.config_loader import ConfigLoader
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class CloudUploader:
    """Classe pour télécharger le modèle vers le cloud (HuggingFace Hub et S3)."""
    
    @staticmethod
    def push_to_huggingface_hub(model, tokenizer, repo_name: str, auth_token: str, message: str = "Initial commit") -> None:
        """
        Télécharge le modèle et le tokenizer sur Hugging Face Hub.

        Args:
            model: Le modèle entraîné.
            tokenizer: Le tokenizer associé.
            repo_name: Nom du dépôt sur Hugging Face Hub.
            auth_token: Jeton d'authentification Hugging Face.
        """
        try:
            # Authentification auprès de Hugging Face Hub
            login(token=auth_token)
            print(f"✅ Connecté à Hugging Face Hub avec succès.")

            # Pousser le tokenizer sur Hugging Face Hub
            print(f"Téléchargement du tokenizer vers Hugging Face Hub : {repo_name}")
            tokenizer.push_to_hub(repo_name, private=False, commit_message=message, use_auth_token=True)
            print(f"✅ Tokenizer téléchargé avec succès : {repo_name}")

            # Pousser le modèle sur Hugging Face Hub
            print(f"Téléchargement du modèle vers Hugging Face Hub : {repo_name}")
            model.push_to_hub(repo_name, private=False, commit_message=message, use_auth_token=True)
            print(f"✅ Modèle téléchargé avec succès : {repo_name}")

        except Exception as e:
            print(f"Erreur lors du téléchargement vers Hugging Face Hub : {e}")
            raise CustomException(f"Erreur lors du téléchargement vers Hugging Face Hub : {e}", e)


    @staticmethod
    def upload_to_google_drive(local_dir: str, google_drive_folder_id: str, google_credentials_path: str,  archive_name: str) -> None:
        """
        Télécharge les fichiers du modèle vers Google Drive.
        
        Args:
            archive_name = "modele_backup.tar.gz"
            folder_id = 
            local_dir: Chemin local du modèle.
            google_drive_folder_id: ID du dossier Google Drive où télécharger les fichiers.
            google_credentials_path: Chemin vers le fichier de configuration OAuth2 pour Google Drive.
        """

        # Créer une archive .tar.gz contenant tous les fichiers du dossier
        logging.info(f"Compression de {local_dir} en {archive_name}...")
        with tarfile.open(archive_name, "w:gz") as archive:
            archive.add(local_dir, arcname=os.path.basename(local_dir))

        logging.info(f"Compression terminée : {archive_name}")
        try:
            # Authentification avec Google Drive
            gauth = GoogleAuth()
            gauth.LoadClientConfigFile(google_credentials_path)  # Charger les identifiants OAuth2
            
            # Vérifier si les identifiants sont valides
            if not gauth.credentials:
                 gauth.LocalWebserverAuth()  # Lancer l'authentification via un navigateur web
            
            drive = GoogleDrive(gauth)
            logging.info(f"Téléchargement des fichiers du modèle vers Google Drive (ID du dossier: {google_drive_folder_id})")
            
            # Liste des fichiers spécifiques à télécharger
            """important_files = ["model.safetensors", "tokenizer.json"]
            for filename in os.listdir(local_dir):
                if filename in important_files:
                    local_file_path = os.path.join(local_dir, filename)"""
                    
            # Créer un fichier sur Google Drive
            file_metadata = {
                        'title': archive_name,
                        'parents': [{'id': google_drive_folder_id}]  # Spécifier le dossier cible
                    }
            file_to_upload = drive.CreateFile(file_metadata)
            file_to_upload.SetContentFile(archive_name)
                    
            # Télécharger le fichier
            logging.info(f"Téléchargement de {archive_name} vers Google Drive")
            file_to_upload.Upload()        
            
            logging.info("Fichiers du modèle téléchargés avec succès vers Google Drive")
        
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement vers Google Drive: {e}")
            raise CustomException(f"Erreur lors du téléchargement vers Google Drive: {e}", e)
        
if __name__ == "__main__":
    # Exemple d'utilisation de la classe
    cloud_uploader = CloudUploader()
    config = ConfigLoader.load_config(config_path="src/finetuning/config/save_params.yaml")
    """params = config.get('google_params', {})
    model_local = params.get("local_model_folder")
    archive_name = params.get("modele_name_compresse")
    google_drive_folder_id = params.get("google_drive_folder_id")
    google_credentials_path = "src/finetuning/config/client_secret_532013038967-9rbvsrd411kb84amrhj0ud74qc2ecs82.apps.googleusercontent.com.json"  
    cloud_uploader.upload_to_google_drive(local_dir=model_local,google_drive_folder_id=google_drive_folder_id , google_credentials_path=google_credentials_path, archive_name=archive_name)"""
    
    params1 = config.get('hugging_face_params', {})
    repo_name = params1.get("depot_name")
    access_token = params1.get("access_token")
    print(repo_name)
    print(access_token)
    cloud_uploader.push_to_huggingface_hub(model=None, tokenizer=None, repo_name=repo_name, auth_token=access_token, message="Fine-tuning Llama 3B sur dialogues Coffria, LoRA")   
   