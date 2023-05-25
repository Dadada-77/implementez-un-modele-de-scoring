import os
import numpy as np
import pandas as pd
import pickle
import string

MODELS_PATH = "Marchand_David_1_dashboard_et_API_052023/backend/ressources/models/"
DATA_PATH = "Marchand_David_1_dashboard_et_API_052023/backend/ressources/data/"
NON_FEATURES = ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'level_0']
TARGET = "TARGET"

def test_simulate_client(input_dict):
    MIN = 0.0
    MAX = 1.0
    output_dict = {}
    clientdf = pd.DataFrame(input_dict)
    features = [f for f in clientdf.columns if f not in NON_FEATURES]

    model = pickle.load(open(MODELS_PATH + "final_model_cpu.pkl", "rb"))
    t = float(np.random.uniform(MIN, MAX))

    probs = model.predict_proba(clientdf[features])
    probs = probs[:, 1]
    pred = int((probs >= t).astype('int')[0])
    probs = float(probs)
    output_dict["PREDICTION"] = [probs, t, pred]

    assert type(probs) == float
    assert probs <= 1.0
    assert probs >= 0.0

    assert type(pred) == int

    assert type(t) == float
    assert t >= 0.0
    assert t <= 1.0

    assert type(output_dict) == dict
    assert len(output_dict["PREDICTION"]) == 3
