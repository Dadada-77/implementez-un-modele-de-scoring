import os,sys
sys.path.append(os.path.abspath("tools"))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from pydantic import BaseModel
from simulation_tools import simulate_client
from visualization_tools import visualize_client_global, roc_model_stats, shap_global_model_stats, return_shap_values, return_global_stats
from generic_tools import return_ids_list, return_data_per_id

class User_input(BaseModel):
    request_type : str
    data : dict

app = FastAPI()

@app.post("/return_data")
def return_data(input:User_input):
    return return_data_per_id(input.request_type, input.data)

@app.post("/return_ids")
def return_ids(input:User_input):
    return return_ids_list(input.request_type)

@app.post("/client_simulation")
def simulate(input:User_input):
    return simulate_client(input.data)

@app.post("/client_global_visualization")
def visualize(input:User_input):
    return visualize_client_global(input.data)

@app.post("/model_stats_global")
def return_stats(input:User_input):
    return return_global_stats()

@app.post("/model_stats_shap")
def return_stats(input:User_input):
    return return_shap_values()

@app.post("/model_stats_roc", response_class=HTMLResponse)
def return_stats(input:User_input):
    return '"""' + roc_model_stats() + '"""'

@app.post("/contact_us")
def contact(input:User_input):
    return "Nous contacter"
