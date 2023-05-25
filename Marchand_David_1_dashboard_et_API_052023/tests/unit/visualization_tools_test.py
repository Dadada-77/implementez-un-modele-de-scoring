import pytest

import pickle

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
import plotly.tools as tls

DATA_PATH = "Marchand_David_1_dashboard_et_API_052023/backend/ressources/data/"
MODELS_PATH = "Marchand_David_1_dashboard_et_API_052023/backend/ressources/models/"
TN = "tn"
TP = "tp"
FN = "fn"
FP = "fp"

def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def test_import_data():
    output_train_df = pd.read_csv(DATA_PATH + 'train_data.csv')
    output_test_df = pd.read_csv(DATA_PATH + 'test_data.csv')

    assert type(output_train_df) == pd.DataFrame
    assert type(output_test_df) == pd.DataFrame

    assert output_train_df.shape[0] > 0
    assert output_train_df.shape[1] > 0
    assert output_test_df.shape[0] > 0
    assert output_test_df.shape[1] > 0

def test_roc_model_stats():
    # Je suppose l'existence du fichier precomputed_roc.pkl, sa création demandant un temps considérable
    results = pickle.load(open(MODELS_PATH + "precomputed_roc.pkl", "rb"))

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

    assert type(html_fig) == str

def test_return_row_predict_state(true_y, pred_y):
    assert type(true_y) == list
    assert type(pred_y) == list

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

    assert type(output) == list
    assert len(output) == len(true_y)

def test_return_main_state(states_list):
    states_dict = {TP : 0,
                   TN : 0,
                   FP : 0,
                   FN : 0}
    for state in (states_list):
        states_dict[state] += 1
    main_state = max(states_dict, key=states_dict.get)

    assert main_state in [TN, TP, FN, FP]
