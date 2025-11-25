"""Train a decision tree classifier on the Palmer penguins dataset.

This script loads the penguins dataset, performs the necessary preprocessing,
trains a decision tree classifier and serialises the trained model along with
the preprocessing artefacts for later use in the Flask API.
"""

import os
import pickle
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

from utils import load_penguins, preprocess


def main() -> None:
    # Use the local CSV because seaborn requires external connectivity.
    csv_path = Path(__file__).resolve().parent.parent / 'data' / 'penguins_size.csv'
    df = load_penguins(use_csv=True, csv_path=str(csv_path))
    X_train, X_test, y_train, y_test, dv, scaler, species_mapping_inv = preprocess(df)

    # Instantiate and fit the decision tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    models_dir = Path(__file__).resolve().parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'tree_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump((dv, scaler, species_mapping_inv, model), f)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()