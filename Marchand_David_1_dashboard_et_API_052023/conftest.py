import pytest
import numpy as np
import pandas as pd

AUTHORIZED_DF_NAMES = ["main_dataframe_pretreated", "newclients_data", "score_data", "test_data"]
DATA_PATH = "./backend/ressources/data/"
ID_COLUMN_NAME = "SK_ID_CURR"
Y_LEN = np.random.randint(100,500)

@pytest.fixture
def df_name():
    input = str(np.random.choice(AUTHORIZED_DF_NAMES))
    return input

@pytest.fixture
def input_dict():
    MIN = -99999999
    MAX = 99999999
    ID_MIN = 0
    input = pd.DataFrame({'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN' : float(np.random.uniform(MIN, MAX)),
                          'BURO_AMT_CREDIT_SUM_MEAN' : float(np.random.uniform(MIN, MAX)),
                          'BURO_DAYS_CREDIT_ENDDATE_MAX' : float(np.random.uniform(MIN, MAX)),
                          'BURO_DAYS_CREDIT_MAX' : float(np.random.uniform(MIN, MAX)),
                          'DAYS_BIRTH' : float(np.random.uniform(MIN, MAX)),
                          'DAYS_EMPLOYED' : float(np.random.uniform(MIN, MAX)),
                          'DAYS_ID_PUBLISH' : float(np.random.uniform(MIN, MAX)),
                          'DAYS_LAST_PHONE_CHANGE' : float(np.random.uniform(MIN, MAX)),
                          'DAYS_REGISTRATION' : float(np.random.uniform(MIN, MAX)),
                          'EXT_SOURCE_1' : float(np.random.uniform(MIN, MAX)),
                          'EXT_SOURCE_2' : float(np.random.uniform(MIN, MAX)),
                          'EXT_SOURCE_3' : float(np.random.uniform(MIN, MAX)),
                          'OWN_CAR_AGE' : float(np.random.uniform(MIN, MAX)),
                          'SK_ID_CURR' : int(np.random.randint(ID_MIN, MAX)),
                          'AMT_CREDIT_TO_ANNUITY_RATIO' : float(np.random.uniform(MIN, MAX)),
                          'AMT_GOODS_PRICE_TO_ANNUITY_RATIO' : float(np.random.uniform(MIN, MAX)),
                          'AMT_CREDIT_GOODS_PRICE_DIFF' : float(np.random.uniform(MIN, MAX)),
                          'GOODS_PRICE_FAIL_RATIO' : float(np.random.uniform(MIN, MAX)),
                          'CREDIT_FAIL_RATIO' : float(np.random.uniform(MIN, MAX))}, index=[0])
    return input.to_dict()

# Simulation du déséquilibre de classe actuellement présent dans les jeux de données !

@pytest.fixture
def true_y():
    return np.random.choice([0, 1], size=(Y_LEN,), p=[0.9, 0.1]).tolist()

@pytest.fixture
def pred_y():
    return np.random.choice([0, 1], size=(Y_LEN,), p=[0.9, 0.1]).tolist()

@pytest.fixture
def states_list():
    input = []
    for i in range(Y_LEN):
        input.append(str(np.random.choice(["tn", "tp", "fn", "fp"])))
    return input
