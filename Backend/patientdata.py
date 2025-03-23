import csv
import json

# Read CSV and convert to JSON
with open('PCOS_DATASET_AUGMENTED_WITH_BMI.csv', 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    data = [row for row in csv_reader]

# Save as JSON
with open('patient_data.json', 'w') as json_file:
    json.dump(data, json_file)