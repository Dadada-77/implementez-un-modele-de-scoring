import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import hydralit_components as hc
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from bytesbufio import BytesBufferIO as BytesIO
from PIL import Image
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import plotly.graph_objects as go
import io
import json
import requests
import pickle
import random
from math import exp
from math import pi
import shap

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Constantes principales
font_size=11
RESSOURCES_PATH = "./ressources/assets/"
SHARED_PATH = "../shared/"
ID_COLUMN_NAME = "SK_ID_CURR"

# Constante à passer à True pour une instance de l'API en local ; False dans le cas d'un conteneur Docker
LOCAL_DEBUG = False

# Boutons princpaux de Streamlit à cacher pour conserver une interface épurée
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --------------------- PARTIE FONCTIONS -------------------------- #

def decision_gauge_plot(input_client_value, input_decision_threshold, input_decision_value):
    """ Dessine un graphe sous la forme d'une jauge, et indiquant la position du client par rapport aux valeurs de réussite / échec de la prédiction """
    input_client_value = round(input_client_value, 2)
    input_decision_threshold = round(input_decision_threshold, 2)

    if(input_decision_value == 0):
        decision_text = "Résultat négatif pour la simulation"
        delta = {'reference': input_decision_threshold, 'increasing': {'color': "red"}}
    elif(input_decision_value == 1):
        decision_text = "Résultat positif pour la simulation"
        delta = {'reference': input_decision_threshold, 'decreasing': {'color': "green"}}

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = input_client_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': decision_text, 'font': {'size': 24}},
        delta = delta,
        gauge = {'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                 'bar': {'color': 'rgba(255,255,255,0.0)'},
                 'bgcolor': "white",
                 'borderwidth': 0,
                 'bordercolor': "gray",
                 'steps': [{'range': [0, input_decision_threshold], 'color': 'green'},
                           {'range': [input_decision_threshold, 100], 'color': 'red'}],
                 'threshold': {'line': {'color': "white", 'width': 4},
                               'thickness': 0.75,
                               'value': input_client_value}}))
    fig.update_layout(height=300, width=800)
    st.plotly_chart(fig, use_container_width=True)

def create_new_client_data(input_counter):
    """ Fonction dédiée à la création d'un nouveau client de zéro """
    col1, col2 = st.columns(2)
    with col1:
        amt_credit_goods_price_diff = st.number_input("Veuillez saisir la valeur de **AMT_CREDIT_GOODS_PRICE_DIFF**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        amt_credit_to_annuity_ratio = st.number_input("Veuillez saisir la valeur de **AMT_CREDIT_TO_ANNUITY_RATIO**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        amt_goods_price_to_annuity_ratio = st.number_input("Veuillez saisir la valeur de **AMT_GOODS_PRICE_TO_ANNUITY_RATIO**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        buro_amt_credit_max_overdue_mean = st.number_input("Veuillez saisir la valeur de **BURO_AMT_CREDIT_MAX_OVERDUE_MEAN**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        buro_amt_credit_sum_mean = st.number_input("Veuillez saisir la valeur de **BURO_AMT_CREDIT_SUM_MEAN**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        buro_days_credit_enddate_max = st.number_input("Veuillez saisir la valeur de **BURO_DAYS_CREDIT_ENDDATE_MAX**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        buro_days_credit_max = st.number_input("Veuillez saisir la valeur de **BURO_DAYS_CREDIT_MAX**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        credit_fail_ratio = st.number_input("Veuillez saisir la valeur de **CREDIT_FAIL_RATIO**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        days_birth = st.number_input("Veuillez saisir la valeur de **DAYS_BIRTH**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
    with col2:
        days_employed = st.number_input("Veuillez saisir la valeur de **DAYS_EMPLOYED**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        days_id_publish = st.number_input("Veuillez saisir la valeur de **DAYS_ID_PUBLISH**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        days_last_phone_change = st.number_input("Veuillez saisir la valeur de **DAYS_LAST_PHONE_CHANGE**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        days_registration = st.number_input("Veuillez saisir la valeur de **DAYS_REGISTRATION**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        ext_source_1 = st.number_input("Veuillez saisir la valeur de **EXT_SOURCE_1**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        ext_source_2 = st.number_input("Veuillez saisir la valeur de **EXT_SOURCE_2**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        ext_source_3 = st.number_input("Veuillez saisir la valeur de **EXT_SOURCE_3**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        goods_price_fail_ratio = st.number_input("Veuillez saisir la valeur de **GOODS_PRICE_FAIL_RATIO**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
        own_car_age = st.number_input("Veuillez saisir la valeur de **OWN_CAR_AGE**", np.nan, step=1e-6, format="%.6f", key=input_counter)
        input_counter += 1
    last_id = send_to_api("return_ids", "newclients_data" , {}, "json")[-1] + 1
    output_df = pd.DataFrame({'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN' : float(buro_amt_credit_max_overdue_mean),
                              'BURO_AMT_CREDIT_SUM_MEAN' : float(buro_amt_credit_sum_mean),
                              'BURO_DAYS_CREDIT_ENDDATE_MAX' : float(buro_days_credit_enddate_max),
                              'BURO_DAYS_CREDIT_MAX' : float(buro_days_credit_max),
                              'DAYS_BIRTH' : float(days_birth),
                              'DAYS_EMPLOYED' : float(days_employed),
                              'DAYS_ID_PUBLISH' : float(days_id_publish),
                              'DAYS_LAST_PHONE_CHANGE' : float(days_last_phone_change),
                              'DAYS_REGISTRATION' : float(days_registration),
                              'EXT_SOURCE_1' : float(ext_source_1),
                              'EXT_SOURCE_2' : float(ext_source_2),
                              'EXT_SOURCE_3' : float(ext_source_3),
                              'OWN_CAR_AGE' : float(own_car_age),
                              'SK_ID_CURR' : int(last_id),
                              'AMT_CREDIT_TO_ANNUITY_RATIO' : float(amt_credit_to_annuity_ratio),
                              'AMT_GOODS_PRICE_TO_ANNUITY_RATIO' : float(amt_goods_price_to_annuity_ratio),
                              'AMT_CREDIT_GOODS_PRICE_DIFF' : float(amt_credit_goods_price_diff),
                              'GOODS_PRICE_FAIL_RATIO' : float(goods_price_fail_ratio),
                              'CREDIT_FAIL_RATIO' : float(credit_fail_ratio)}, index=[0])
    output_df.replace(np.nan, -99999999, inplace=True)
    return output_df, input_counter

def fetch_new_client_data(input_counter):
    """ Renvoi d'un DataFrame pour un ancien client sélectionné """
    clients_ids = send_to_api("return_ids", "newclients_data" , {}, "json")

    st.title("Simulation de prêt ancien client")
    client_id = st.selectbox("Veuillez séléctionner l'identifiant du client à visualiser :", clients_ids, key=input_counter)
    input_counter += 1

    if(client_id != ""):
        output = pd.DataFrame(send_to_api("return_data", "newclients_data" , {ID_COLUMN_NAME : client_id}, "json"))
        return output, input_counter

@st.cache_data(persist=True, show_spinner=False)
def send_to_api(endpoint, specific_request_type, input_data, response_format):
    """ Envoi de la requête vers le Backend et récupération de la réponse """
    tosend = {"request_type" : specific_request_type, "data" : input_data}
    if(LOCAL_DEBUG == True):
        received = requests.post(url = "http://127.0.0.1:8000/" + endpoint, data = json.dumps(tosend), timeout=8000)
    else:
        received = requests.post(url = "http://host.docker.internal:8000/" + endpoint, data = json.dumps(tosend), timeout=8000)

    if (response_format == "json"):
        return json.loads(received.text)
    else:
        return received.text

def draw_decision_bars(input_features, input_counter, input_dict):
    """ Dessine un ensemble de barres par variable, reflétant l'état de prédiction du client par variable """
    override_theme_1 = {'content_color': '#68DC8F', 'progress_color' : '#68DC8F'}
    override_theme_2 = {'content_color': '#E9DC8E', 'progress_color' : '#E9DC8E'}
    override_theme_3 = {'content_color': '#D6AA77', 'progress_color' : '#D6AA77'}
    override_theme_4 = {'content_color': '#D66577', 'progress_color' : '#D66577'}

    for feature in (input_features):
        if(input_dict["VARIABLE"]["STATE"][feature] == "tp"):
            feature_override_theme = override_theme_1
            bar_size = 100
            state_str = "TP, " + str(feature)
        elif(input_dict["VARIABLE"]["STATE"][feature] == "fp"):
            feature_override_theme = override_theme_2
            bar_size = 75
            state_str = "FP, " + str(feature)
        elif(input_dict["VARIABLE"]["STATE"][feature] == "fn"):
            feature_override_theme = override_theme_3
            bar_size = 50
            state_str = "FN, " + str(feature)
        elif(input_dict["VARIABLE"]["STATE"][feature] == "tn"):
            feature_override_theme = override_theme_4
            bar_size = 25
            state_str = "TN, " + str(feature)
        st.write('<p style="font-size:' + str(font_size) + 'px;">' + str(state_str) + '</p>',unsafe_allow_html=True)
        hc.progress_bar(bar_size, '-', override_theme=feature_override_theme, key=input_counter)
        input_counter+=1
    return input_counter

def compute_mean_color(input_states):
    """ Renvoi de la couleur correspondant à l'état principal retrouvé dans la liste fournie en entrée """
    max_value = max(input_states, key=input_states.get)
    color_dict = {'tp' : '#68DC8F', 'tn' : '#D66577', 'fp' : '#E9DC8E', 'fn' : '#D6AA77'}
    return color_dict[max_value]

def return_radar_plot_values(input_features, input_dict):
    """ Renvoie une liste composée des valeurs et de la couleur majoritaire à fournir pour la fonction draw_radar_plot """
    output_values = {}
    states_dict = {"tp" : 0, "tn" : 0, "fp" : 0, "fn" : 0}
    values_dict = {"tp" : 4, "tn" : 1, "fp" : 3, "fn" : 2}
    for feature in (input_features):
        states_dict[input_dict["VARIABLE"]["STATE"][feature]] += 1
        output_values[feature] = values_dict[input_dict["VARIABLE"]["STATE"][feature]]
    output_color = compute_mean_color(states_dict)
    return [output_values, output_color]

def draw_radar_plot(input_features, input_dict):
    """ Affiche un radar plot représentant les états du client par variable """
    r_theta_color = return_radar_plot_values(input_features, input_dict)
    r_list = [i for i in r_theta_color[0].values()]
    theta_list = [i for i in r_theta_color[0].keys()]
    radar_color = r_theta_color[1]

    fig = go.Figure(data=go.Scatterpolar(r=r_list, theta=theta_list, fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True),),showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

def define_clients_range_per_variable(input_data, input_counter, input_feature):
    """ Fonction dédiée à l'affinement des clients par variable """
    adapt_range = st.selectbox("", ["", "Affiner la gamme de clients"], key=input_counter)
    input_counter += 1
    if(adapt_range != ""):
        adapt_values = st.slider("Gamme de valeurs acceptées", 0.0, 1.0, (0.0, 1.0), 0.05)
        send_box = st.selectbox("", ["", "Affiner !"], key=input_counter)
        input_counter += 1
        if(send_box != ""):
            send_data = {"DATA" : input_data, "RANGE" : {input_feature : adapt_values}}
            viz_received = send_to_api("client_global_visualization", "", send_data, "json")
            variables_states = viz_received["VARIABLE"]["STATE"]
            features = tuple([i for i in viz_received["VARIABLE"]["IMAGE"].keys()])
    return input_counter

def import_shap_values(exported_dict):
    """ Import d'une SHAP value via le dictionnaire fourni par la fonction export_shap_values """
    output = shap.Explanation(values=np.array(exported_dict["VALUES"]),
                              base_values=np.array(exported_dict["BASE_VALUES"]),
                              data=np.array(exported_dict["DATA"]),
                              feature_names=exported_dict["FEATURE_NAMES"])
    return output

def show_api_results(input_data, input_counter):
    """ Fonction permettant l'affichage des divers résultats de visualisation """
    send_data = {"DATA" : input_data, "RANGE" : {}}
    viz_received = send_to_api("client_global_visualization", "", send_data, "json")
    variables_states = viz_received["VARIABLE"]["STATE"]
    features = tuple([i for i in viz_received["VARIABLE"]["IMAGE"].keys()])
    features_1 = features[:len(features)//2]
    features_2 = features[len(features)//2:]

    col4, col5, col6 = st.columns((2, 0.25, 2))
    with col4:
        selected_decision_visualization = st.selectbox("Veuillez séléctionner la méthode de visualisation des prises de décision par variable :", ["", "Barres", "Radar Plot"], key=input_counter)
        input_counter += 1
        if(selected_decision_visualization == "Barres"):
            col4_1, col4_2 = st.columns((1, 1))

            with col4_1:
                input_counter = draw_decision_bars(features_1, input_counter, viz_received)

            with col4_2:
                input_counter = draw_decision_bars(features_2, input_counter, viz_received)

        elif(selected_decision_visualization == "Radar Plot"):
            col4_3, col4_4, col4_5 = st.columns((0.1,1,0.1))
            with col4_4:
                draw_radar_plot(list(features), viz_received)

    with col6:
        features = list(features)
        features.insert(0, "")
        features.insert(len(features)+1, "Interprétabilité locale du modèle")
        features.insert(len(features)+1, "Interprétabilité globale du modèle")
        selected_feature_1 = st.selectbox("Veuillez séléctionner la distribution de la variable à visualiser :", features, key=input_counter)
        input_counter += 1
        if (selected_feature_1 != ""):
            if (selected_feature_1 == "Interprétabilité locale du modèle"):
                shap_values = import_shap_values(viz_received["SHAP_CLIENT"])
                st.pyplot(shap.plots.beeswarm(shap_values))
            elif (selected_feature_1 == "Interprétabilité globale du modèle"):
                shap_values = import_shap_values(viz_received["SHAP_ALL"])
                st.pyplot(shap.plots.beeswarm(shap_values))
            else:
                plt.clf()
                fig1, ax1 = plt.subplots()
                plt.grid(False)
                plt.axis('off')
                plt.imshow(np.reshape(viz_received["VARIABLE"]["IMAGE"][selected_feature_1], (400, 800, 3)))
                st.pyplot(fig1)
                input_counter = define_clients_range_per_variable(input_data, input_counter, selected_feature_1)

        selected_feature_2 = st.selectbox("Veuillez séléctionner la distribution de la variable à visualiser :", features, key=input_counter)
        input_counter += 1
        if (selected_feature_2 != ""):
            if (selected_feature_2 == "Interprétabilité locale du modèle"):
                shap_values = import_shap_values(viz_received["SHAP_CLIENT"])
                st.pyplot(shap.plots.beeswarm(shap_values))
            elif (selected_feature_2 == "Interprétabilité globale du modèle"):
                shap_values = import_shap_values(viz_received["SHAP_ALL"])
                st.pyplot(shap.plots.beeswarm(shap_values))
            else:
                plt.clf()
                fig2, ax2 = plt.subplots()
                plt.grid(False)
                plt.axis('off')
                plt.imshow(np.reshape(viz_received["VARIABLE"]["IMAGE"][selected_feature_2], (400, 800, 3)))
                st.pyplot(fig2)
                input_counter = define_clients_range_per_variable(input_data, input_counter, selected_feature_2)
    return input_counter

def display_main_menu():
    """ Fonction proposant l'affichage du menu latéral """
    with st.sidebar:
        menu_list = ["Accueil", "Nouveau Client", "Ancien Client", "Modèle de Simulation", "Nous contacter"]
        menu_name = "Simulateur de Prêt"
        icons_list = ['house-fill', 'person-plus-fill', 'person-fill', 'bar-chart-fill', 'send-plus-fill']
        menu_icon = "currency-euro"
        menu_style = {"container": {"padding": "5!important", "background-color": "#fafafa"},
                      "icon": {"color": "black", "font-size": "25px"},
                      "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                      "nav-link-selected": {"background-color": "#ABCDEF"},}

        output_choose = option_menu(menu_name, menu_list, icons=icons_list, menu_icon=menu_icon, default_index=0, styles=menu_style)

        st.sidebar.image(RESSOURCES_PATH + "company_logo.png", use_column_width=True)
        return output_choose

def welcome_page():
    """ Page de bienvenue sur l'application """
    st.title('Bienvenue sur le simulateur de prêt de Prêt à Dépenser !')
    st.image(RESSOURCES_PATH + 'welcome.png')

def new_client_treatment(input_counter):
    """ Page dédiée aux nouveaux clients """
    st.title("Simulation de prêt nouveau client")
    main_selection = st.selectbox("Vous pouvez choisir de saisir un nouveau client, ou alors en sélectionner un déjà saisi", ["<Sélectionnez>", "Saisie nouveau client", "Sélection nouveau client déjà inscrit"], key=input_counter)
    input_counter += 1
    if(main_selection != ""):
        if(main_selection == "Saisie nouveau client"):
            clientdf, input_counter = create_new_client_data(input_counter)
        elif(main_selection == "Sélection nouveau client déjà inscrit"):
            clientdf, input_counter = fetch_new_client_data(input_counter)
    submitted = st.selectbox("Veuillez séléctionner l'action à réaliser :", ["", "Envoyer"], key=input_counter)
    input_counter += 1
    if(submitted != ""):
        received = send_to_api("client_simulation", "" , clientdf.to_dict(), "json")
        client_value = (1 - received["PREDICTION"][0]) * 100
        threshold_value = (1 - received["PREDICTION"][1]) * 100
        decision_value = received["PREDICTION"][2]
        decision_gauge_plot(client_value, threshold_value, decision_value)
        input_counter = show_api_results(clientdf.to_dict(), input_counter)
    return input_counter

def old_client_treatment(input_counter):
    """ Page dédiée aux anciens clients """
    old_clients_ids = send_to_api("return_ids", "test_data" , {}, "json")

    st.title("Simulation de prêt ancien client")
    client_id = st.selectbox("Veuillez séléctionner l'identifiant du client à visualiser :", old_clients_ids, key=input_counter)
    input_counter += 1

    if(client_id != ""):
        clientdf = pd.DataFrame(send_to_api("return_data", "test_data" , {ID_COLUMN_NAME : client_id}, "json"))
        st.dataframe(clientdf)
        received = send_to_api("client_simulation", "", clientdf.to_dict(), "json")
        client_value = (1 - received["PREDICTION"][0]) * 100
        threshold_value = (1 - received["PREDICTION"][1]) * 100
        decision_value = received["PREDICTION"][2]
        decision_gauge_plot(client_value, threshold_value, decision_value)
        input_counter = show_api_results(clientdf.to_dict(), input_counter)
    return input_counter

def roc_plot():
    """ Affichage d'une courbe ROC """
    received = send_to_api("model_stats_roc", "", {}, "html")
    html.html(received[3:-3], height=800)

def shap_global_plot_bar():
    """ Affichage de l'interprétabilité globale """
    shap_values = send_to_api("model_stats_shap", "", {}, "json")
    st.pyplot(shap.plots.bar(import_shap_values(shap_values)))

def shap_global_plot_beeswarm():
    """ Affichage de l'interprétabilité globale / locale """
    shap_values = send_to_api("model_stats_shap", "", {}, "json")
    st.pyplot(shap.plots.beeswarm(import_shap_values(shap_values)))

def general_stats():
    """ Affichage des statistiques générales """
    general_stats_df = pd.DataFrame(send_to_api("model_stats_global", "", {}, "json"))
    st.dataframe(general_stats_df)

def model_stats():
    """ Page dédiée à l'affichage des statistiques du modèle """
    st.title("Modèle de Simulation")
    # Sélectionnez est présent afin d'avoir un placeholder le temps de sélectionner une vraie statistique sélectionnée
    model_stats_list = ('<Sélectionnez>', 'Statistiques Générales', 'Courbe ROC', "Interprétabilité Globale (Bar)", "Interprétabilité Globale (Beeswarm)")

    displayed_model_stat = st.selectbox('Choisissez la statistique du modèle à visualiser', model_stats_list)

    if (displayed_model_stat == model_stats_list[1]): # Statistiques Générales
        general_stats()
    elif (displayed_model_stat == model_stats_list[2]): # Courbe ROC
        roc_plot()
    elif (displayed_model_stat == model_stats_list[3]): # Interprétabilité Globale (Bar)
        shap_global_plot_bar()
    elif (displayed_model_stat == model_stats_list[4]): # Interprétabilité Globale (Beeswarm)
        shap_global_plot_beeswarm()

def contact_us():
    """ Page dédiée au contact """
    st.title("Nous contacter")
    with st.form(key='columns_in_form2',clear_on_submit=True):
        st.write("Veuillez utiliser ce formulaire en cas de soucis avec l'application")

        contact_name = st.text_input(label='Saisissez votre Nom')
        contact_email = st.text_input(label='Saisissez votre adresse e-mail')
        contact_message = st.text_input(label='Saisissez votre message. Décrivez de la façon la plus précise le souci rencontré')

        submitted = st.form_submit_button('Envoyer')

        if submitted:
            send = json.dumps([choose, [contact_name, contact_email, contact_message]])
            requests.post(url = "http://127.0.0.1:8000/simulateur_pret", data = send)
            st.write('Votre message a bien été envoyé. Nous vous répondrons dans les plus brefs délais. Merci !')

def main():
    """ Fonction principale """
    key_counter = 0

    # En premier, j'affiche un menu sur la gauche
    choose = display_main_menu()

    # En second, l'affichage principal va différer selon le choix de l'utilisateur au niveau du menu
    if choose == "Accueil":
        welcome_page()

    elif choose == "Nouveau Client":
        key_counter = new_client_treatment(key_counter)

    elif choose == "Ancien Client":
        key_counter = old_client_treatment(key_counter)

    elif choose == "Modèle de Simulation":
        key_counter = model_stats()

    elif choose == "Nous contacter":
        contact_us()

if __name__ == '__main__':
    main()
