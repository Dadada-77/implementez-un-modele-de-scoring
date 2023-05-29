import os
import numpy as np
import pandas as pd
import pickle
import string

MODELS_PATH = "./ressources/models/"
DATA_PATH = "./ressources/data/"

def simulate_client(input_dict):
    """ Retourne la prédiction du modèle retenu pour un dataframe d'entrée """
    output_dict = {}
    clientdf = pd.DataFrame(input_dict)

    non_features = ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'level_0']
    features = [f for f in clientdf.columns if f not in non_features]
    target = "TARGET"

    model = pickle.load(open(MODELS_PATH + "final_model.pkl", "rb"))
    t = 0.1

    probs = model.predict_proba(clientdf[features])
    probs = probs[:, 1]
    pred = int((probs >= t).astype('int')[0])
    probs = float(probs)

    output_dict["PREDICTION"] = [probs, t, pred]

    return output_dict
