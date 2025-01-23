from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load pre-trained model and scaler
model = load_model('ev_battery_degradation_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # This is the home page

@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    try:
        if request.is_json:
            data = request.get_json()

            # Extract and convert features
            try:
                features = [float(data.get(f'feature_{i}')) for i in range(1, 15)]
            except ValueError:
                return jsonify({"error": "Invalid input. Ensure all features are numeric"}), 400

            # Check for missing values
            if None in features:
                return jsonify({"error": "All 14 features must be provided"}), 400

            # Convert to numpy array and reshape
            feature_array = np.array(features).reshape(1, -1)

            # Apply scaling if necessary
            scaled_features = scaler.transform(feature_array)

            # Predict using the model
            prediction = model.predict(scaled_features)

            # Convert result to JSON-friendly format
            predicted_health = float(prediction[0][0])

            # Return prediction
            return jsonify({'prediction': predicted_health})

        else:
            return jsonify({"error": "Request must be in JSON format"}), 415

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": "An error occurred during prediction", "details": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
