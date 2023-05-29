# Implémentez un modèle de scoring - Partie Backend

Cette sous-partie reprend la partie Backend de mon application proposée à l'occasion du Projet 7 d'OpenClassrooms. Elle se charge de délivrer les réponses souhaitées par le Frontend au travers d'une API développée via le Package FastAPI.  
  
Le Backend s'architecture ainsi :  
  - backend.py : le script amorçant le backend  
  - tools  
     - generic_tools.py : fonctions génériques  
     - simulation_tools.py : fonctions liées à la prédiction  
     - visualization_tools.py : fonctions liées à la visualisation  
  - ressources  
     - data : dossier comprenant les différents dataframes repris dans la note méthodologique présente à la racine du dépôt  
     - evidently : dossier comprenant les rapports de Data Drift d'Evidently  
     - mlruns : dossier comprenant le tracking des expérimentations de ML via MLFlow  
     - models : dossier comprenant les différents modèles (actuellement sérialisés par Pickle, prochainement stockés via MLFlow)  
