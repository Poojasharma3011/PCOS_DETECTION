from fastapi import FastAPI
import numpy as np
from model import load_model, predict_pcos

app = FastAPI()

# Load the trained Q-learning model
q_table, X = load_model()

@app.post("/predict_pcos")
def predict_pcos_api(patient_data: dict):
    return predict_pcos(q_table, X, patient_data)

@app.get("/")
def home():
    return {"message": "PCOS Detection API is running"}

