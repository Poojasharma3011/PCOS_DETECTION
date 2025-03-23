from flask import Flask, render_template, request
import requests

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/predict_pcos"  # FastAPI Backend URL

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Collect user input
        patient_data = {
            "Age": float(request.form['Age']),
            "Weight": float(request.form['Weight']),
            "Height": float(request.form['Height']),
            "BMI": float(request.form['BMI']),
            "Blood Group": float(request.form['Blood_Group']),
            "Cycle": float(request.form['Cycle_Regularity']),
            "Cycle Length": float(request.form['Period_Length']),
            "Weight Gain": float(request.form['Weight_Gain']),
            "Facial Hair": float(request.form['Facial_Hair']),
            "Skin Darkening": float(request.form['Skin_Darkening']),
            "Hair Loss": float(request.form['Hair_Loss']),
            "Acne": float(request.form['Acne']),
            "Fast Food": float(request.form['Fast_Food']),
            "Regular Exercise": float(request.form['Regular_Exercise']),
            "Mood Swings": float(request.form['Mood_Swings']),
            "Follicle No. (R)": float(request.form['Follicle_NoR']), 
            "Follicle No. (L)": float(request.form['Follicle_NoL']),
            "TSH (mIU/L)": float(request.form['TSH']) 
        }
        
        # Send data to FastAPI for prediction
        response = requests.post(FASTAPI_URL, json=patient_data)
        prediction = response.json()
        
        return render_template('index.html', prediction=prediction)
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
