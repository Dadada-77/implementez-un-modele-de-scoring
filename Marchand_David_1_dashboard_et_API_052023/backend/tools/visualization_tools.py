import os
import pickle
import gc

import numpy as np
import pandas as pd

import xgboost as xgb
from tqdm.notebook import tqdm

import plotly.graph_objects as go
import plotly.io as pio

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedKFold
from imblearn.under_sampling import RandomUnderSampler

import mlflow

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constantes dédiées au chargement des chemins

MLFLOW_PATH = "./ressources/mlruns/"
MODELS_PATH = "./ressources/models/"
DATA_PATH = "./ressources/data/"

# Saisir ici l'identifiant du modèle retenu dans MLFlow

FINAL_MODEL_EXP_ID = "6352986c44714f79b6d360e1e5a84ae6"

# Constantes dédiées à Matplotlib

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Constantes prévues pour le traitement de l'état de la prédiction

TP, TN, FP, FN = ('tp', 'tn', 'fp', 'fn')
colors_states = {TP : '#68DC8F', TN : '#D66577', FP : '#E9DC8E', FN : '#D6AA77'}

# --------------------- PARTIE FONCTIONS -------------------------- #

def to_labels(pos_probs, threshold):
    """ Permet de renvoyer les prédictions en format binaire (plus précisément en entier) """
    return (pos_probs >= threshold).astype('int')

def shap_global_model_stats():
    """ Renvoie les SHAP Values précalculées """
    return pickle.load(open(MODELS_PATH + "final_model_shap_values.pkl", "rb"))

def import_data_nan(input_path):
    """ Renvoie un DataFrame avec des valeurs manquantes sous la forme np.nan """
    OUTLIER_VALUE = -99999999

    output_df = pd.read_csv(input_path)
    output_df.replace(OUTLIER_VALUE, np.nan, inplace=True)

    return output_df

def import_data_outlier(input_path):
    """ Renvoie un DataFrame avec des valeurs manquantes sous la forme OUTLIER_VALUE """
    OUTLIER_VALUE = -99999999

    output_df = pd.read_csv(input_path)
    output_df.replace(np.nan, OUTLIER_VALUE, inplace=True)

    return output_df

def roc_pr_compute():
    """ Calcul des résultats nécessaires à la création d'un graphe de type ROC / PR AUC """
    if(os.path.exists(MODELS_PATH + "precomputed_roc.pkl") == False):
        train_df = import_data_outlier(DATA_PATH + "train_data.csv")
        test_df = import_data_outlier(DATA_PATH + "test_data.csv")

        non_features = ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'level_0']
        features = [f for f in train_df.columns if f not in non_features]
        target = "TARGET"

        params = {'objective'          : 'binary:logistic',
                  'colsample_bytree'   : 0.933,
                  'enable_categorical' : False,
                  'gamma'              : 1,
                  'max_depth'          : 3,
                  'min_child_weight'   : 1070,
                  'n_estimators'       : 100,
                  'reg_alpha'          : 0.03,
                  'reg_lambda'         : 0.05,
                  'subsample'          : 1}

        X_train = train_df[features]
        X_test = test_df[features]
        y_train = train_df[target]
        y_test = test_df[target]

        cv    = RepeatedKFold(n_splits=5, n_repeats=10, random_state=101)
        folds = [(train,test) for train, test in cv.split(X_train, y_train)]

        metrics = ['auc', 'fpr', 'tpr', 'thresholds']
        results = {'train': {m:[] for m in metrics},
                   'val'  : {m:[] for m in metrics},
                   'test' : {m:[] for m in metrics}}

        dtest = xgb.DMatrix(X_test, label=y_test)
        for train, test in tqdm(folds, total=len(folds)):
            dtrain = xgb.DMatrix(X_train.iloc[train,:], label=y_train.iloc[train])
            dval   = xgb.DMatrix(X_train.iloc[test,:], label=y_train.iloc[test])
            model  = xgb.train(
                dtrain                = dtrain,
                params                = params,
                evals                 = [(dtrain, 'train'), (dval, 'val')],
                num_boost_round       = 1000,
                verbose_eval          = False,
                early_stopping_rounds = 10,
            )
            sets = [dtrain, dval, dtest]
            for i,ds in enumerate(results.keys()):
                y_preds              = model.predict(sets[i])
                labels               = sets[i].get_label()
                fpr, tpr, thresholds = roc_curve(labels, y_preds)
                results[ds]['fpr'].append(fpr)
                results[ds]['tpr'].append(tpr)
                results[ds]['thresholds'].append(thresholds)
                results[ds]['auc'].append(roc_auc_score(labels, y_preds))
        pickle.dump(results, open(MODELS_PATH + "precomputed_roc.pkl", "wb"))
    else:
        results = pickle.load(open(MODELS_PATH + "precomputed_roc.pkl", "rb"))

    return results

def roc_model_stats():
    """ Création d'un graphe ROC AUC """
    results = roc_pr_compute()

    kind = 'val'
    c_fill      = 'rgba(52, 152, 219, 0.2)'
    c_line      = 'rgba(52, 152, 219, 0.5)'
    c_line_main = 'rgba(41, 128, 185, 1.0)'
    c_grid      = 'rgba(189, 195, 199, 0.5)'
    c_annot     = 'rgba(149, 165, 166, 0.5)'
    c_highlight = 'rgba(192, 57, 43, 1.0)'
    fpr_mean    = np.linspace(0, 1, 100)
    interp_tprs = []
    for i in range(10):
        fpr           = results[kind]['fpr'][i]
        tpr           = results[kind]['tpr'][i]
        interp_tpr    = np.interp(fpr_mean, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)
    tpr_mean     = np.mean(interp_tprs, axis=0)
    tpr_mean[-1] = 1.0
    tpr_std      = 2*np.std(interp_tprs, axis=0)
    tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
    tpr_lower    = tpr_mean-tpr_std
    auc          = np.mean(results[kind]['auc'])
    fig = go.Figure([
        go.Scatter(
            x          = fpr_mean,
            y          = tpr_upper,
            line       = dict(color=c_line, width=1),
            showlegend = False,
            name       = 'upper'),
        go.Scatter(
            x          = fpr_mean,
            y          = tpr_lower,
            fill       = 'tonexty',
            fillcolor  = c_fill,
            line       = dict(color=c_line, width=1),
            showlegend = False,
            name       = 'lower'),
        go.Scatter(
            x          = fpr_mean,
            y          = tpr_mean,
            line       = dict(color=c_line_main, width=2),
            showlegend = True,
            name       = f'AUC: {auc:.3f}')
    ])
    fig.add_shape(
        type ='line',
        line =dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_layout(
        template    = 'plotly_white',
        title_x     = 0.5,
        xaxis_title = "Specificity",
        yaxis_title = "Sensitivity",
        width       = 800,
        height      = 800,
        legend      = dict(
            yanchor="bottom",
            xanchor="right",
            x=0.95,
            y=0.01,
        )
    )
    fig.update_yaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        scaleanchor = "x",
        scaleratio  = 1,
        linecolor   = 'black')
    fig.update_xaxes(
        range       = [0, 1],
        gridcolor   = c_grid,
        constrain   = 'domain',
        linecolor   = 'black')

    html_fig = pio.to_html(fig)
    return html_fig

def return_global_stats():
    """ Renvoie les statistiques du modèle final stocké dans MLFlow """
    # Je vais charger les métriques stockées par MLFlow
    mlflow.set_tracking_uri(MLFLOW_PATH)
    mlruns = mlflow.search_runs()
    final_model_logs = mlruns[mlruns.loc[:, "run_id"] == FINAL_MODEL_EXP_ID]
    final_model_stats_df = pd.DataFrame({"Accuracy" : final_model_logs["metrics.Accuracy"].values[0],
                                         "FBeta Score" : final_model_logs["metrics.Best FBeta Score"].values[0],
                                         "Training time" : final_model_logs["metrics.Entraînement du Fine Tuned XGBoost model"].values[0],
                                         "Prediction time" : final_model_logs["metrics.Prédiction du fine_tuned_XGBoost model"].values[0],
                                         "Business Score" : final_model_logs["metrics.Business Scoring value"].values[0],
                                         "ROC AUC" : final_model_logs["metrics.ROC AUC"].values[0],
                                         "Precision Recall AUC" : final_model_logs["metrics.Precision Recall Curve AUC"].values[0],
                                         "Macro AVG F1-Score" : final_model_logs["metrics.Macro AVG F1-Score"].values[0],
                                         "Macro AVG Precision" : final_model_logs["metrics.Macro AVG Precision"].values[0],
                                         "Macro AVG Recall" : final_model_logs["metrics.Macro AVG Recall"].values[0]}, index=[0])
    return final_model_stats_df.to_dict()

def return_row_predict_state(true_y, pred_y):
    """ Renvoie une liste composée des états de prédiction (Vrais / Faux Positifs / Négatifs) """
    output = []
    for i in range (len(true_y)):
        if (true_y[i] == pred_y[i] and true_y[i] == 0):
            output.append(TN)
        elif (true_y[i] == pred_y[i] and true_y[i] == 1):
            output.append(TP)
        elif (true_y[i] != pred_y[i] and true_y[i] == 1):
            output.append(FP)
        elif (true_y[i] != pred_y[i] and true_y[i] == 0):
            output.append(FN)
        else:
            output.append(np.nan)
    return output

def export_shap_values(input_shap_value):
    """ Export d'une SHAP Value dans un format accepté par FastAPI """
    output_dict = {}
    output_dict["VALUES"] = input_shap_value.values.tolist()
    output_dict["BASE_VALUES"] = input_shap_value.base_values.tolist()
    output_dict["DATA"] = input_shap_value.data.tolist()
    output_dict["FEATURE_NAMES"] = input_shap_value.feature_names
    return output_dict

def return_shap_values():
    """ Renvoie les SHAP Values globales """
    shap_global_values = pickle.load(open(MODELS_PATH + "final_model_shap_values.pkl", "rb"))
    return export_shap_values(shap_global_values)

def return_confusion_matrix_data():
    """ Retourne les données nécessaires à la création d'un matrice de confusion """
    output_dict = {"TRUE_Y" : "", "PRED_Y" : "", "CLASSES" : ""}

    testdf = import_data_nan(DATA_PATH + "test_data.csv")
    
    model = pickle.load(open(MODELS_PATH + "final_model.pkl", "rb"))
    t = 0.1
    
    non_features = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
    features = [f for f in testdf.columns if f not in non_features]
    target = "TARGET"
    
    probs = model.predict_proba(testdf[features])
    probs = probs[:, 1]
    testdf["PREDICTION"] = to_labels(probs, t)
    
    output_dict["TRUE_Y"] = testdf[target].tolist()
    output_dict["PRED_Y"] = testdf["PREDICTION"].tolist()
    output_dict["CLASSES"] = model.classes_.tolist()
    
    return output_dict

def visualize_client_global(input_dict):
    """ Fonction renvoyant les données nécessaires pour l'établissement de graphes d'états """
    trimmed_variable_dict = input_dict["RANGE"]
    input_dict = input_dict["DATA"]
    output_dict = {"DATA" : "", "SHAP_ALL" : "", "SHAP_LOCAL" : "", "VARIABLE" : {}}
    
    clientdf = pd.DataFrame(input_dict)
    testdf = import_data_nan(DATA_PATH + "test_data.csv")
    
    if(trimmed_variable_dict != {}):
        target_variable = list(trimmed_variable_dict.keys())[0]
        min_value = trimmed_variable_dict[target_variable][0]
        max_value = trimmed_variable_dict[target_variable][1]
        testdf = testdf[(testdf.loc[:, target_variable] > min_value) & (testdf.loc[:, target_variable] < max_value)]
        testdf.drop(["level_0"], axis=1, inplace=True)
        testdf.reset_index(inplace=True)
        
    model = pickle.load(open(MODELS_PATH + "final_model.pkl", "rb"))
    t = 0.1
    
    non_features = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']
    features = [f for f in testdf.columns if f not in non_features]
    target = "TARGET"

    probs = model.predict_proba(testdf[features])
    probs = probs[:, 1]
    testdf["PREDICTION"] = to_labels(probs, t)
    testdf['PREDICTION_STATE'] = return_row_predict_state(testdf[target], testdf['PREDICTION'])

    undersample = RandomUnderSampler(sampling_strategy='auto')
    X, y = undersample.fit_resample(testdf[features], testdf['PREDICTION_STATE'])
    unsmpled_df = X
    unsmpled_df['PREDICTION_STATE'] = y
    unsmpled_df.reset_index(inplace=True)

    non_plotted_feats = ['index', 'level_0', 'PREDICTION_STATE']
    plotted_feats = [f for f in unsmpled_df.columns if f not in non_plotted_feats]
    
    unsmpled_df.replace(np.nan, -99999999, inplace=True)
    clientdf.replace(np.nan, -99999999, inplace=True)
    
    # Retour des SHAP values locales
    shap_explainer = pickle.load(open(MODELS_PATH + "final_model_shap_explainer.pkl", "rb"))
    shap_local_values = shap_explainer(clientdf[features])
    output_dict["SHAP_CLIENT"] = export_shap_values(shap_local_values)

    # Retour des SHAP values globales
    shap_global_values = pickle.load(open(MODELS_PATH + "final_model_shap_values.pkl", "rb"))
    output_dict["SHAP_ALL"] = export_shap_values(shap_global_values)
    
    output_dict["DATA"] = unsmpled_df.to_dict()
    
    # Nettoyage !
    del clientdf
    del testdf
    del unsmpled_df

    gc.collect()

    return output_dict
