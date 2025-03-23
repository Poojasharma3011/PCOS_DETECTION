import numpy as np
import pickle

def load_model():
    with open("pcos_model.pkl", "rb") as file:
        q_table, X = pickle.load(file)
    return q_table, X

def predict_pcos(q_table, X, patient_data):
    features = np.array([patient_data[key] for key in patient_data])
    best_action = np.argmax(q_table[0])  # Predict using Q-table
    return {"prediction": "positive" if best_action == 1 else "negative"}