import joblib
from flask import Flask, request, jsonify

# تحميل الموديل
model = joblib.load("random_forest_model_34class.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "No features provided"}), 400
    prediction = model.predict([features])
    return jsonify({"result": str(prediction[0])})
