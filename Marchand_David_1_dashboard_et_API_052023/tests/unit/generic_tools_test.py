import pytest
import numpy as np
import pandas as pd

AUTHORIZED_DF_NAMES = ["main_dataframe_pretreated", "newclients_data", "score_data", "test_data", "train_data"]
DATA_PATH = "Marchand_David_1_dashboard_et_API_052023/backend/ressources/data/"
ID_COLUMN_NAME = "SK_ID_CURR"

def test_return_ids_list(df_name):
    df = pd.read_csv(DATA_PATH + df_name + ".csv")
    output_ids = sorted(df[ID_COLUMN_NAME].unique().tolist())
    output_ids.insert(0, "")

    assert type(df_name) == str
    assert df_name in AUTHORIZED_DF_NAMES
    assert type(output_ids) == list
    assert len(output_ids) > 0

def test_return_data_per_id(df_name, row_id=None):
    df = pd.read_csv(DATA_PATH + df_name + ".csv")
    min_id = min(df[ID_COLUMN_NAME])
    max_id = max(df[ID_COLUMN_NAME])
    row_id = {ID_COLUMN_NAME : np.random.randint(min_id, max_id)}

    output = df[df.loc[:, ID_COLUMN_NAME] == row_id[ID_COLUMN_NAME]].to_dict()

    assert type(output) == dict
    assert len(output) > 0
