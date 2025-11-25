"""Train a logistic regression classifier on the Palmer penguins dataset.

This script loads the penguins dataset, preprocesses it (dropping rows with
missing values, standardising numeric features and one‑hot encoding
categoricals), trains a multi‑class logistic regression model and serialises
the resulting model alongside the preprocessing artefacts.  It also prints
basic accuracy metrics on the training and test sets.
"""

import os
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from utils import load_penguins, preprocess


def main() -> None:
    # Load and preprocess the data.  Fall back to the bundled CSV because
    # loading via seaborn requires internet access which may not be available.
    csv_path = Path(__file__).resolve().parent.parent / 'data' / 'penguins_size.csv'
    df = load_penguins(use_csv=True, csv_path=str(csv_path))
    X_train, X_test, y_train, y_test, dv, scaler, species_mapping_inv = preprocess(df)

    # Instantiate and fit the logistic regression model
    model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=500)
    model.fit(X_train, y_train)

    # Report simple accuracy metrics
    train_acc = (model.predict(X_train) == y_train).mean()
    test_acc = (model.predict(X_test) == y_test).mean()
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Ensure the models directory exists one level up from this script
    models_dir = Path(__file__).resolve().parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)

    # Serialise the vectoriser, scaler, inverse mapping and model together
    model_path = models_dir / 'logreg_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump((dv, scaler, species_mapping_inv, model), f)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()