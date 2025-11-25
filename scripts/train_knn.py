"""Train a k‑nearest neighbours classifier on the Palmer penguins dataset.

This script loads and preprocesses the dataset, trains a KNN classifier
with a reasonable default neighbourhood size, then saves the trained model
alongside all preprocessing components for later inference.
"""

import os
import pickle
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

from utils import load_penguins, preprocess


def main() -> None:
    # Use the bundled CSV file because seaborn's online dataset is not accessible.
    csv_path = Path(__file__).resolve().parent.parent / 'data' / 'penguins_size.csv'
    df = load_penguins(use_csv=True, csv_path=str(csv_path))
    X_train, X_test, y_train, y_test, dv, scaler, species_mapping_inv = preprocess(df)

    # Use 5 neighbours as a reasonable default for this size of dataset
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    models_dir = Path(__file__).resolve().parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'knn_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump((dv, scaler, species_mapping_inv, model), f)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()