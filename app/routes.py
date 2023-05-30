from flask import request, jsonify, render_template
from app import app 
import pickle
import numpy as np


modelo_ml = pickle.load(open("app/models/eater.pkl", "rb"))

def convertCardToFeatures(card):
    return [card]

@app.route('/predict', methods=['POST'])
def predict_page():
    if not request.form or 'card' not in request.form:
        return "La solicitud debe contener un campo 'card'.", 400

    features = convertCardToFeatures(request.form['card'])

    prediction = modelo_ml.predict(np.array(features).reshape(1, -1))

    return render_template('result.html', prediction=bool(prediction[0]))
if __name__ == "__main__":
    test_bins = ['5154620012345678', '5154620098765432']
    for test_bin in test_bins:
        features = convertCardToFeatures(test_bin)
        prediction = modelo_ml.predict(np.array(features).reshape(1, -1))
        print(f'Bin: {test_bin}, Prediction: {bool(prediction[0])}')