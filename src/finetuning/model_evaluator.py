import torch
from typing import Any, Dict, Optional
from src.finetuning.logger import logging
from src.finetuning.exception import CustomException
from unsloth import (
    FastLanguageModel
)


class ModelEvaluator:
    """Classe pour évaluer les performances du modèle affiné."""
    
    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialise l'évaluateur de modèle.
        
        Args:
            model: Le modèle à évaluer
            tokenizer: Le tokenizer du modèle
        """
        self.model = model
        self.tokenizer = tokenizer
        logging.info("ModelEvaluator initialisé")
    
    def prepare_for_inference(self) -> Any:
        """
        Prépare le modèle pour l'inférence.
        
        Returns:
            Le modèle préparé pour l'inférence
        """
        try:
            logging.info("Préparation du modèle pour l'inférence")
            inference_model = FastLanguageModel.for_inference(self.model)
            logging.info("Modèle préparé pour l'inférence")
            return inference_model
        
        except Exception as e:
            logging.error(f"Erreur lors de la préparation pour l'inférence: {e}")
            raise
    
    def generate_response(self, inference_model: Any, prompt: str, langue: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            inference_model: Le modèle préparé pour l'inférence
            prompt: Le prompt pour générer une réponse
            generation_params: Paramètres de génération
            
        Returns:
            La réponse générée
        """
        try:
            if generation_params is None:
                generation_params = {
                    "max_new_tokens": 512,
                    "use_cache": True,
                    "temperature": 0.7,
                    "min_p": 0.1
                }
            
            logging.info(f"Génération d'une réponse pour le prompt: {prompt[:50]}...")
            

            if langue == "fr":
                 instruction_prompt = "Vous êtes un assistant IA spécialisé en recherche documentaire sur la plateforme Coffria. Votre rôle est d’aider les utilisateurs à trouver des documents pertinents en analysant et affinant leur requête. Vous affinez la recherche en fonction de critères spécifiques comme les mots-clés, la date de publication, le type de document, la source de confiance, et d’autres éléments pertinents. Vous vous exprimez uniquement en français, de manière claire, concise et respectueuse."
            elif langue == "en":
                 instruction_prompt = "You are an AI assistant specialized in document research on the Coffria platform. Your role is to assist users in finding relevant documents by analyzing and refining their queries. You refine the search based on specific criteria such as keywords, publication date, document type, trusted source, and other relevant factors. You communicate exclusively in English, in a clear, concise, and respectful manner."

            # Formatage du message
            instruction =  f"{instruction_prompt}"
            messages = [{"role": "system", "content": instruction},
                        {"role": "user", "content": prompt}]
 
            # Tokenisation de l'entrée
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Création du masque d'attention
            attention_mask = (inputs != self.tokenizer.pad_token_id).long()
            
            # Génération de la sortie
            outputs = inference_model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                **generation_params
            )
            
            # Décodage de la sortie
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Post-traitement pour nettoyer les messages système indésirables
            clean_output = []
            for response in decoded_output:
                response = response.split("user\n\n")[1] if "user\n\n" in response else response
                response = response.split("assistant\n\n")[1] if "assistant\n\n" in response else response
                clean_output.append(response)
            
            logging.info("Réponse générée avec succès")
            return clean_output[0]
        
        except Exception as e:
            logging.error(f"Erreur lors de la génération de réponse: {e}")
            raise CustomException(f"Erreur lors de la génération de réponse: {e}", e)
