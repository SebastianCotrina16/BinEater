from flask import request, jsonify
from app import app
import pickle
import numpy as np


modelo_ml = pickle.load(open("models/eater.pkl", "rb"))

@app.route('/api/predict', methods=['POST'])
def predict():
    
    if not request.json or 'card' not in request.json:
        return jsonify({'error': 'La solicitud debe ser un objeto JSON y debe contener un campo "card".'}), 400

    #TODO CONVERTCARDTOFEATURES
    features = convertCardToFeatures(request.json['card'])

    prediction = modelo_ml.predict(np.array(features).reshape(1, -1))

    return jsonify({'fraudulent': bool(prediction[0])})
