from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'completion_predictor.pkl')
MODEL_PIPELINE = None

####################### Model Pipeline #######################
def load_model_pipeline():
    """Loads the pre-trained model pipeline."""
    global MODEL_PIPELINE
    try:
        if os.path.exists(MODEL_PATH):
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            print(f"Model pipeline loaded successfully from {MODEL_PATH}")
        else:
            print(f"Error: Model pipeline file not found at {MODEL_PATH}. Prediction endpoint will not work.")
            MODEL_PIPELINE = None
    except Exception as e:
        print(f"Error loading model pipeline: {e}")
        MODEL_PIPELINE = None

####################### API Endpoints #######################
@app.route('/', methods=['GET'])
def home():
    """Home endpoint to check if the API is running."""
    return jsonify({"message": "Video Completion Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict_completion():
    """Predicts if a learner will complete a video based on input features."""
    if MODEL_PIPELINE is None:
        return jsonify({"error": "Model pipeline is not loaded. Cannot make predictions."}), 503 # Service Unavailable

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    print(f"Received data: {data}")

    required_features = ['learner_avg_completion_rate', 'course_code']
    if not all(feature in data for feature in required_features):
        missing = [f for f in required_features if f not in data]
        return jsonify({"error": f"Missing required features: {missing}"}), 400

    try:
        avg_completion_rate = data['learner_avg_completion_rate']
        if not isinstance(avg_completion_rate, (int, float)):
            raise ValueError("Must be a number.")
        if not (0 <= avg_completion_rate <= 100):
            return jsonify({
                "error": "Invalid value for 'learner_avg_completion_rate'. Must be between 0 and 100 (inclusive)."
            }), 400
    except (ValueError, TypeError) as e:
         return jsonify({
            "error": f"Invalid data type or value for 'learner_avg_completion_rate'. {e}"
         }), 400

    if not isinstance(data['course_code'], str):
        return jsonify({"error": "'course_code' must be a string."}), 400

    try:
        input_df = pd.DataFrame([data], columns=required_features)
        prediction = MODEL_PIPELINE.predict(input_df)[0]
        probability = MODEL_PIPELINE.predict_proba(input_df)[0]

        prediction_label = "Will Complete" if int(prediction) == 1 else "Will Not Complete"
        probability_complete = float(probability[1])
        probability_not_complete = float(probability[0])

        return jsonify({
            "prediction": int(prediction),
            "prediction_label": prediction_label,
            "probability_complete": probability_complete,
            "probability_not_complete": probability_not_complete,
            "message": "Prediction successful."
        })

    except ValueError as ve:
         print(f"Error during prediction (possibly unknown category): {ve}")
         return jsonify({"error": f"Prediction failed. Check input data format or values. Details: {ve}"}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed due to an internal server error."}), 500

####################### Main Execution #######################
if __name__ == '__main__':
    load_model_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=True)
