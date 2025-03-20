# 1. Callback simplifié avec vérification complète
import json
import os
from transformers import TrainerCallback

class GuaranteedLossLogger(TrainerCallback):
    def __init__(self, log_file="training_logs.json"):
        self.log_file = log_file
        # Initialiser le fichier de logs s'il n'existe pas
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump([], f)  # Créer une liste vide pour les logs

    def on_step_end(self, args, state, control, **kwargs):
        # Vérifier si log_history contient des éléments et si 'loss' est présent
        if state.log_history and len(state.log_history) > 0:
            last_log = state.log_history[-1]
            if 'loss' in last_log:
                print(f"Step {state.global_step}: Loss = {last_log['loss']}")
                # Sauvegarder les logs dans le fichier en mode ajout
                self._save_log(last_log)
            else:
                print(f"Step {state.global_step}: Aucune perte disponible dans les logs.")
        else:
            print(f"Step {state.global_step}: Aucun historique de logs disponible.")

    def _save_log(self, log_entry):
        # Lire les logs existants
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []  # Si le fichier est vide ou corrompu, initialiser une liste vide

        # Ajouter le nouvel élément
        logs.append(log_entry)

        # Réécrire les logs dans le fichier
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4, ensure_ascii=False)