from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Load the trained model and encoders
model = joblib.load('oa_drug_response_model.pkl')
le_gender = joblib.load('gender_label_encoder.pkl')
le_drug_type = joblib.load('drug_type_label_encoder.pkl')
le_dosage_level = joblib.load('dosage_level_label_encoder.pkl')
le_activity_level = joblib.load('activity_level_label_encoder.pkl')
le_smoking_status = joblib.load('smoking_status_label_encoder.pkl')
le_alcohol_consumption = joblib.load('alcohol_consumption_label_encoder.pkl')
le_response = joblib.load('response_label_encoder.pkl')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Get inputs
        age = float(data["age"])
        gender = le_gender.transform([data["gender"]])[0]
        bmi = float(data["bmi"])
        oa_severity = float(data["oa_severity"])
        duration_of_oa = int(data["duration_of_oa"])
        crp = float(data["crp"])
        esr = float(data["esr"])
        drug_type = le_drug_type.transform([data["drug_type"]])[0]
        dosage_level = le_dosage_level.transform([data["dosage_level"]])[0]
        treatment_duration = int(data["treatment_duration"])
        activity_level = le_activity_level.transform([data["activity_level"]])[0]
        diet_score = float(data["diet_score"])
        smoking_status = le_smoking_status.transform([data["smoking_status"]])[0]
        alcohol_consumption = le_alcohol_consumption.transform([data["alcohol_consumption"]])[0]

        # Prepare input
        input_data = np.array([[age, gender, bmi, oa_severity, duration_of_oa, crp, esr,
                                drug_type, dosage_level, treatment_duration, activity_level,
                                diet_score, smoking_status, alcohol_consumption]])

        # Make prediction
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = le_response.inverse_transform([prediction_encoded])[0]

        return jsonify({"prediction": prediction_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Choose any port or use default 10000


