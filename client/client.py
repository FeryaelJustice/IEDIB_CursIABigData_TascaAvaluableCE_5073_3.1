"""Simple client to test the penguin species classification API.

This script makes HTTP POST requests to each of the available model endpoints
with a couple of sample penguin measurements.  It prints the JSON
responses so you can see the predicted class and probabilities.
"""

import json
import requests


def main() -> None:
    base_url = 'http://127.0.0.1:5000/predict/'
    # Two example individuals from the dataset
    samples = [
        {
            "island": "Biscoe",
            "sex": "Male",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
        },
        {
            "island": "Dream",
            "sex": "Female",
            "bill_length_mm": 36.5,
            "bill_depth_mm": 17.4,
            "flipper_length_mm": 186.0,
            "body_mass_g": 3625.0,
        },
    ]
    models = ['logreg', 'svm', 'tree', 'knn']
    for model_name in models:
        print(f"\nTesting model: {model_name}")
        for idx, sample in enumerate(samples, start=1):
            url = base_url + model_name
            try:
                response = requests.post(url, json=sample, timeout=3)
                response.raise_for_status()
                print(f"Sample {idx} response: {json.dumps(response.json(), indent=2)}")
            except Exception:
                # If the server is not reachable, fall back to local prediction
                print(f"Sample {idx} using local model (API unreachable)")
                # Lazy import of pickle and preprocessing helpers
                import pickle
                from pathlib import Path
                # Load model artefacts
                model_path = Path(__file__).resolve().parent.parent / 'models' / f"{model_name}_model.pkl"
                with open(model_path, 'rb') as f:
                    dv, scaler, species_mapping_inv, model = pickle.load(f)
                # Apply the same preprocessing as the server
                num_cols = scaler['num_cols']
                mean = scaler['mean']
                std = scaler['std']
                transformed = sample.copy()
                for col in num_cols:
                    transformed[col] = (float(transformed[col]) - mean[col]) / std[col]
                X = dv.transform([transformed])
                probs = model.predict_proba(X)[0]
                pred_idx = int(probs.argmax())
                species = species_mapping_inv[pred_idx]
                print(json.dumps({'class': pred_idx, 'species': species, 'probabilities': probs.tolist()}, indent=2))


if __name__ == '__main__':
    main()