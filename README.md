README.txt

# Affinage Modulaire du Modèle Llama 3.2 3B pour Coffria

## 📌 Introduction

Ce projet fournit un script Python modulaire conçu pour affiner le modèle pré-entraîné Llama 3.2 3B. L'objectif principal est d'améliorer la capacité du modèle à traiter des dialogues en français, spécifiquement pour les tâches de recherche documentaire sur la plateforme Coffria. Coffria est une plateforme dédiée à la recherche sécurisée de documents professionnels. Ce script facilite l'adaptation du modèle pour gérer des requêtes et des résultats liés à des documents académiques, des études de marché et des rapports, optimisant ainsi son utilisation dans un contexte professionnel francophone.

## ⚙️ Prérequis Techniques

Avant de commencer, assurez-vous que votre environnement dispose des prérequis suivants :

- Python 3.10 ou supérieur
- Pip (gestionnaire de paquets Python)
- GPU NVIDIA compatible CUDA (recommandé pour l'affinage, CPU possible pour des tests limités)
- Librairies Python listées dans le fichier `requirements.txt` (voir ci-dessous)

Fichier `requirements.txt`:

```text
torch --index-url https://download.pytorch.org/whl/cu118
torchvision --index-url https://download.pytorch.org/whl/cu118
torchaudio --index-url https://download.pytorch.org/whl/cu118
xformers[torch2]
unsloth[colab] @ git+https://github.com/unslothai/unsloth.git
git+https://github.com/huggingface/transformers.git
trl
boto3
pydrive
google-api-python-client
huggingface_hub
-e .

🚀 Installation et Utilisation
1. Cloner le dépôt GitHub :

git clone [URL_DU_DÉPÔT]
cd [NOM_DU_RÉPERTOIRE_DU_DÉPÔT]

2. Créer et activer un environnement virtuel (recommandé) :

python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows

3. Installer les dépendances :

pip install -r requirements.txt

4. Configuration :

Fichiers YAML : Le script utilise des fichiers YAML pour la configuration. Assurez-vous que les fichiers model_loading_params.yaml, lora_params.yaml et trainer_params.yaml sont correctement configurés (voir section "🔧 Configuration").

Dataset : Préparez votre jeu de données de dialogues en français et anglais au format JSON, ou modifiez le script pour utiliser un dataset Hugging Face existant. Modifiez le chemin vers votre dataset dans trainer_params.yaml ou directement dans le script principal (fine_tuning.py) si nécessaire.

AWS (Optionnel) : Si vous souhaitez télécharger le modèle vers AWS S3, configurez vos identifiants AWS dans aws_config.yaml.

5. Lancer l'affinage :

Exécutez le script principal fine_tuning.py :

python -m src.finetuning.fine_tuning

Le script chargera le modèle, préparera le dataset, lancera l'affinage, sauvegardera le modèle affiné et (optionnellement) le téléchargera vers Hugging Face Hub et S3, selon votre configuration.

🔧 Guide de Configuration
La configuration du projet est gérée via des fichiers YAML pour une modularité accrue :

- model_loading_params.yaml : Paramètres de Chargement du Modèle

model_name: "unsloth/Llama-3.2-3B-Instruct"  # Nom du modèle Hugging Face
quantization_bits: 4                     # Bits pour la quantification (4 ou 8)
device: "cuda"                           # 'cuda' ou 'cpu'
max_seq_length: 2048                     # Longueur maximale des séquences

- lora_params.yaml : Paramètres LoRA (Low-Rank Adaptation)

r: 16                # Rang LoRA
lora_alpha: 32       # Alpha LoRA
lora_dropout: 0.05   # Dropout LoRA
target_modules:      # Modules cibles pour LoRA
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

- trainer_params.yaml : Paramètres de l'Entraîneur (Trainer)  

dataset_path: "path/dialogues_dataset.json"    # Chemin vers votre dataset JSON
output_dir: "llama-3b-coffria-french"          # Répertoire de sortie pour le modèle affiné
per_device_train_batch_size: 4                 # Batch size par GPU
gradient_accumulation_steps: 4                 # Étapes d'accumulation de gradient
warmup_steps: 20                               # Étapes de warmup
max_steps: 300                                 # Nombre maximal d'étapes d'entraînement
learning_rate: 0.00015                         # Taux d'apprentissage
weight_decay: 0.01                             # Décroissance du poids
logging_steps: 50                              # Fréquence des logs
num_epochs: 3                                  # Nombre d'époques
dataset_num_proc: 2                            # Nombre de processus pour le dataset

Modifiez ces fichiers YAML pour adapter la configuration à vos besoins spécifiques (modèle, dataset, hyperparamètres, chemins, identifiants AWS).

🛠 Maintenance et Améliorations Possibles
Ce script modulaire fournit une base solide pour l'affinage du modèle Llama 3.2 3B. Voici quelques pistes d'amélioration et de maintenance :

Ajout de métriques d'évaluation : Intégrer des métriques d'évaluation (perplexity, métriques spécifiques à la tâche) pendant et après l'entraînement pour un suivi plus précis des performances du modèle.

Implémentation d'une boucle d'évaluation : Développer une fonction d'évaluation plus complète pour tester le modèle affiné sur un jeu de données de test dédié et mesurer ses performances sur des tâches spécifiques à Coffria.

Recherche d'hyperparamètres : Mettre en place une recherche d'hyperparamètres (via des outils comme Optuna ou Ray Tune) pour optimiser les performances du modèle affiné en explorant différentes configurations d'hyperparamètres LoRA et d'entraînement.

Support pour d'autres modèles Llama ou modèles de langage : Étendre le script pour supporter l'affinage d'autres modèles de la famille Llama ou d'autres modèles de langage pré-entraînés (en modifiant les classes ModelLoader et potentiellement DatasetPreparation).

Documentation et tests unitaires : Ajouter une documentation plus complète du code (docstrings, commentaires) et développer des tests unitaires pour assurer la robustesse et la qualité du code.

🖊 Contribu