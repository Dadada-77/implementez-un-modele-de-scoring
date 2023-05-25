import os, sys
import io
import re
from bytesbufio import BytesBufferIO as BytesIO
import base64
import numpy as np
import pandas as pd
import pickle
import random
import string
import xgboost as xgb
from tqdm.notebook import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import plotly.tools as tls
import shap

MODELS_PATH = "./ressources/models/"
DATA_PATH = "./ressources/data/"

from sklearn.metrics import fbeta_score, RocCurveDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedKFold
from imblearn.under_sampling import RandomUnderSampler

from starlette.responses import StreamingResponse

import gc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation_tools import simulate_client

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

TP, TN, FP, FN = ('tp', 'tn', 'fp', 'fn')
colors_states = {TP : '#68DC8F', TN : '#D66577', FP : '#E9DC8E', FN : '#D6AA77'}

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def general_model_stats():
    # Load model
    model = pickle.load(open(MODELS_PATH + "final_model_cpu.pkl", "rb"))

def shap_global_model_stats():
    return pickle.load(open(MODELS_PATH + "final_model_shap_values.pkl", "rb"))

def import_data():
    OUTLIER_VALUE = -99999999

    output_train_df = pd.read_csv(DATA_PATH + 'train_data.csv')
    output_test_df = pd.read_csv(DATA_PATH + 'test_data.csv')

    output_train_df.replace(np.nan, OUTLIER_VALUE, inplace=True)
    output_test_df.replace(np.nan, OUTLIER_VALUE, inplace=True)

    return output_train_df, output_test_df

def roc_pr_compute():
    if(os.path.exists(MODELS_PATH + "precomputed_roc.pkl") == False):
        train_df = pd.read_csv(DATA_PATH + "train_data.csv")
        test_df = pd.read_csv(DATA_PATH + "test_data.csv")

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

def precision_recall_model_stats():
    # Results sent are the same, they will differ in how data will be displayed !
    results = roc_pr_compute()

def return_row_predict_state(true_y, pred_y):
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

def return_main_state(states_list):
    states_dict = {TP : 0,
                   TN : 0,
                   FP : 0,
                   FN : 0}
    for state in (states_list):
        states_dict[state] += 1
    return max(states_dict, key=states_dict.get)

def visualize_client_global(input_dict):
    trimmed_variable_dict = input_dict["RANGE"]
    input_dict = input_dict["DATA"]
    clientdf = pd.DataFrame(input_dict)
    testdf = pd.read_csv(DATA_PATH + "test_data.csv")

    clientdf.replace(np.nan, -99999999)
    testdf.replace(np.nan, -99999999)

    non_features = ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index', 'level_0']
    features = [f for f in clientdf.columns if f not in non_features]
    target = "TARGET"

    model = pickle.load(open(MODELS_PATH + "final_model_cpu.pkl", "rb"))
    t = 0.095

    if(trimmed_variable_dict != {}):
        target_variable = list(trimmed_variable_dict.keys())[0]
        min_value = trimmed_variable_dict[target_variable][0]
        max_value = trimmed_variable_dict[target_variable][1]
        testdf = testdf[(testdf.loc[:, target_variable] > min_value) & (testdf.loc[:, target_variable] < max_value)]
        testdf.drop(["level_0"], axis=1, inplace=True)
        testdf.reset_index(inplace=True)

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
    plotted_feats = feats = [f for f in unsmpled_df.columns if f not in non_plotted_feats]

    plt.rcParams["figure.figsize"] = [8, 4]
    output_dict = {"VARIABLE" : {"IMAGE" : {}, "STATE" : {}}, "SHAP_ALL" : "", "SHAP_CLIENT" : ""}

    for feature in (plotted_feats):
        client_feature_state = TN
        fig, ax = plt.subplots()

        N, bins, patches = ax.hist(unsmpled_df[feature], bins=20, edgecolor='white', alpha=0.6, linewidth=1.5)

        for i in range(len(N)):
            start = float(str(patches[i]).split(',')[0].split('=')[1][1:])
            if (i == 0):
                saved_start = start
            end = start + float(str(patches[i]).split(',')[2].split('=')[1])
            feature_state = return_main_state(unsmpled_df['PREDICTION_STATE'][(unsmpled_df.loc[:, feature] >= start) & (unsmpled_df.loc[:, feature] < end)])
            patches[i].set_facecolor(colors_states[return_main_state(unsmpled_df['PREDICTION_STATE'][(unsmpled_df.loc[:, feature] >= start) & (unsmpled_df.loc[:, feature] < end)])])
            if(clientdf[feature].values[0] >= start and clientdf[feature].values[0] < end):
                client_feature_state = feature_state
        plt.axvline(x = clientdf[feature].values[0], color = 'blue', linewidth=3, linestyle = 'dotted')
        plt.xticks(bins, rotation=-45)
        fig.tight_layout()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        output_dict["VARIABLE"]["IMAGE"][feature] = image_from_plot.tolist()
        output_dict["VARIABLE"]["STATE"][feature] = client_feature_state
        plt.close(fig)
        plt.clf()

    # Return global SHAP values
    shap_global_values = pickle.load(open(MODELS_PATH + "final_model_shap_values.pkl", "rb"))

    """
    # Return local SHAP values
    shap_explainer = pickle.load(open(MODELS_PATH + "final_model_shap_values.pkl", "rb"))
    shap_local_values = shap_explainer(clientdf[features])

    output_dict["SHAP_CLIENT"] = shap_local_values
    output_dict["SHAP_ALL"] = shap_global_values.tolist()
    """

    # Cleaning everything

    del clientdf
    del testdf
    del unsmpled_df

    gc.collect()

    # Time to return our data !

    print(output_dict)

    return output_dict
