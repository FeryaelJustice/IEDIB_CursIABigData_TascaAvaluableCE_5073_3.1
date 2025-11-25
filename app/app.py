"""Flask API for penguin species classification.

This application exposes RESTful endpoints for each of the trained models.
Clients can POST a JSON payload containing the penguin's measurements and
categorial descriptors and receive a predicted species along with class
probabilities.

Example payload::

    {
        "island": "Biscoe",
        "sex": "Male",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0
    }

The API will respond with a JSON object containing the predicted class index,
the species name and a list of class probabilities.
"""

import json
import pickle
from pathlib import Path

from flask import Flask, jsonify, request


app = Flask(__name__)


# A dictionary mapping model names to their loaded artefacts
MODELS: dict[str, dict[str, object]] = {}


def load_model(name: str, filename: str) -> None:
    """Load a model and its preprocessing artefacts from a pickle file.

    Parameters
    ----------
    name : str
        A key under which to store the model in the MODELS dictionary.
    filename : str
        Relative filename of the pickle to load from the models directory.
    """
    model_path = Path(__file__).resolve().parent.parent / 'models' / filename
    with open(model_path, 'rb') as f:
        dv, scaler, species_mapping_inv, model = pickle.load(f)
    MODELS[name] = {
        'dv': dv,
        'scaler': scaler,
        'species_mapping_inv': species_mapping_inv,
        'model': model,
    }


def preprocess_input(data: dict, dv, scaler) -> object:
    """Convert a raw JSON payload into a feature array suitable for prediction.

    The numeric values are standardised using the stored mean and standard
    deviation, and the categoricals are left untouched.  The resulting
    dictionary is then transformed by the DictVectorizer into a numpy array.

    Parameters
    ----------
    data : dict
        Input JSON dictionary.
    dv : DictVectorizer
        Fitted vectoriser used to convert dictionaries to arrays.
    scaler : dict
        Dictionary containing ``mean``, ``std`` and ``num_cols`` for scaling.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (1, n_features) ready for the model.
    """
    num_cols = scaler['num_cols']
    mean = scaler['mean']
    std = scaler['std']
    # Copy the input to avoid mutating caller's dictionary
    transformed = data.copy()
    for col in num_cols:
        value = transformed.get(col)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Expected numeric value for {col}, got {value!r}")
        transformed[col] = (value - mean[col]) / std[col]
    # Use the vectoriser to convert into the final feature array
    return dv.transform([transformed])


# Load all models at module import time
load_model('logreg', 'logreg_model.pkl')
load_model('svm', 'svm_model.pkl')
load_model('tree', 'tree_model.pkl')
load_model('knn', 'knn_model.pkl')


@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name: str):
    """Predict the penguin species using the specified model.

    Parameters
    ----------
    model_name : str
        The key of the model to use (``logreg``, ``svm``, ``tree``, ``knn``).

    Returns
    -------
    flask.Response
        A JSON response containing the predicted class, species and class
        probabilities.
    """
    if model_name not in MODELS:
        return jsonify({'error': f"Model '{model_name}' not found"}), 404
    payload = request.get_json() or {}
    model_info = MODELS[model_name]
    dv = model_info['dv']
    scaler = model_info['scaler']
    species_mapping_inv = model_info['species_mapping_inv']
    model = model_info['model']
    try:
        X = preprocess_input(payload, dv, scaler)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    # Compute probabilities and determine the most probable class
    probs = model.predict_proba(X)[0]
    pred_idx = int(probs.argmax())
    species = species_mapping_inv[pred_idx]
    return jsonify({'class': pred_idx, 'species': species, 'probabilities': probs.tolist()})


if __name__ == '__main__':
    # Running with debug=True will auto‑reload on code changes which is handy during development
    app.run(debug=True)