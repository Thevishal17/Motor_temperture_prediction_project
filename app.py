

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("motor-temperature/models/model.pkl")
scaler = joblib.load("motor-temperature/models/scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Validate and get features
        feature_names = ["ambient_temp", "coolant_temp", "torque", "motor_speed"]
        features = []
        for f in feature_names:
            value = request.form.get(f)
            if value is None:
                raise ValueError(f"Missing input: {f}")
            features.append(float(value))
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        return render_template("index.html", prediction_text=f"Predicted Motor Temperature: {prediction[0]:.2f} °C")
    except ValueError as ve:
        return render_template("index.html", prediction_text=f"Input Error: {ve}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
