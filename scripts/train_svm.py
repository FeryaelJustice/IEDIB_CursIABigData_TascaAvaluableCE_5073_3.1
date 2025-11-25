"""Train a support vector machine classifier on the Palmer penguins dataset.

This script loads the dataset, applies the same preprocessing pipeline as
other models, trains a multi‑class SVM with probability estimates enabled and
serialises the fitted model along with preprocessing artefacts.
"""

import os
import pickle
from pathlib import Path

from sklearn.svm import SVC

from utils import load_penguins, preprocess


def main() -> None:
    # Use the bundled CSV because seaborn requires internet access.
    csv_path = Path(__file__).resolve().parent.parent / 'data' / 'penguins_size.csv'
    df = load_penguins(use_csv=True, csv_path=str(csv_path))
    X_train, X_test, y_train, y_test, dv, scaler, species_mapping_inv = preprocess(df)

    # We enable probability estimates to allow predict_proba in the API
    model = SVC(kernel='rbf', probability=True, gamma='scale')
    model.fit(X_train, y_train)

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    models_dir = Path(__file__).resolve().parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'svm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump((dv, scaler, species_mapping_inv, model), f)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()