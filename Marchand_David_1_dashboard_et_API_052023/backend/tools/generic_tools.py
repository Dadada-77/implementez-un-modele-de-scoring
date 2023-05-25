import numpy as np
import pandas as pd

MODELS_PATH = "./ressources/models/"
DATA_PATH = "./ressources/data/"
ID_COLUMN_NAME = "SK_ID_CURR"

def return_ids_list(df_name):
    df = pd.read_csv(DATA_PATH + df_name + ".csv")
    output_ids = sorted(df[ID_COLUMN_NAME].unique().tolist())
    output_ids.insert(0, "")
    return output_ids

def return_data_per_id(df_name, row_id):
    df = pd.read_csv(DATA_PATH + df_name + ".csv")
    return df[df.loc[:, ID_COLUMN_NAME] == row_id[ID_COLUMN_NAME]].to_dict()
