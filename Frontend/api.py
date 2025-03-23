import pandas as pd
import google.generativeai as genai

# Initialize the Gemini AI client (Replace with your actual API key)
client=genai.configure(api_key="AIzaSyDgpz5afI7STp8FKSqoCbCGdapQWyFE7Gk")

# Load test data from CSV
csv_file = "PCOS_DATASET_AUGMENTED_WITH_BMI.csv"  # Replace with your file path
data = pd.read_csv("csv_file")

# Define a function to format patient data into a query
def generate_pcos_prompt(patient):
    return f"""
    A patient has the following health data:
    - Age: {patient['Age']}
    - BMI: {patient['BMI']}
    - Irregular Periods: {patient['Irregular_Periods']}
    - Insulin Resistance: {patient['Insulin_Resistance']}
    - Hair Growth: {patient['Hair_Growth']}
    - Acne: {patient['Acne']}
    - Weight Gain: {patient['Weight_Gain']}
    - Other Symptoms: {patient['Other_Symptoms']}

    Based on this data, assess the likelihood of PCOS and suggest possible next steps.
    """

# Process each patient in the dataset
for index, row in data.iterrows():
    prompt = generate_pcos_prompt(row)
    
    # Get AI response
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    # Print or store the result
    print(f"Patient {index+1}:\n{response.text}\n{'-'*50}")

