import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


def load_penguins(use_csv: bool = False, csv_path: str | None = None) -> pd.DataFrame:
    """Load the Palmer penguins dataset.

    If ``use_csv`` is True and a ``csv_path`` is provided the CSV file is loaded,
    otherwise the dataset is fetched via :func:`seaborn.load_dataset`.  The Kaggle
    version of this dataset uses slightly different column names (``culmen_length_mm``
    and ``culmen_depth_mm`` instead of ``bill_length_mm`` and ``bill_depth_mm``).
    These names are normalised here so downstream code can treat them uniformly.

    Rows with missing values in any column are dropped.

    Parameters
    ----------
    use_csv : bool, optional
        Whether to load the dataset from a CSV file.  Defaults to False.
    csv_path : str, optional
        Path to a CSV file containing the dataset.  Only used when
        ``use_csv`` is True.  Defaults to None.

    Returns
    -------
    pd.DataFrame
        The cleaned penguins dataset.
    """
    if use_csv and csv_path:
        df = pd.read_csv(csv_path)
        # Normalise column names to match the seaborn dataset
        rename_map = {
            'culmen_length_mm': 'bill_length_mm',
            'culmen_depth_mm': 'bill_depth_mm',
        }
        df = df.rename(columns=rename_map)
    else:
        # Using seaborn to fetch the dataset ensures the correct column names
        df = sns.load_dataset("penguins")

    # Drop any rows with missing values for simplicity
    df = df.dropna()
    return df


def preprocess(df: pd.DataFrame):
    """Preprocess the penguins dataset into train and test sets and return
    everything needed for model training.

    This function performs the following steps:

    * Maps the ``species`` column to integer labels and stores the mapping.
    * Splits the dataset into training and test sets (80/20) using a stratified
      split on the labels to preserve class ratios.
    * Computes the mean and standard deviation of the numeric columns on the
      training set and uses these to standardise both training and test sets.
    * One‑hot encodes the categorical variables using a :class:`DictVectorizer`.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned penguins dataset.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test, dv, scaler, species_mapping_inv)``
        where ``dv`` is the fitted DictVectorizer, ``scaler`` contains the
        mean and standard deviation for each numeric column along with the
        column names and ``species_mapping_inv`` maps label integers back
        to species names for easier interpretation of predictions.
    """
    # Determine the mapping from species names to integer labels
    species = sorted(df['species'].unique())
    species_mapping = {name: idx for idx, name in enumerate(species)}
    df['target'] = df['species'].map(species_mapping)
    y = df['target'].values

    # Define which columns are categorical and which are numeric
    cat_cols = ['island', 'sex']
    num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

    # Split into train and test sets using a stratified split on the labels
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

    # Compute mean and std of numeric columns on the training data
    mean = df_train[num_cols].mean()
    std = df_train[num_cols].std()

    # Standardise numeric columns in both train and test using training statistics
    df_train[num_cols] = (df_train[num_cols] - mean) / std
    df_test[num_cols] = (df_test[num_cols] - mean) / std

    # Convert the DataFrame rows to dictionaries for the DictVectorizer
    def df_to_dict(df_part: pd.DataFrame):
        return df_part[cat_cols + num_cols].to_dict(orient='records')

    train_dict = df_to_dict(df_train)
    test_dict = df_to_dict(df_test)

    # Fit a DictVectorizer on the training data and transform both train and test
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)
    X_train = dv.transform(train_dict)
    X_test = dv.transform(test_dict)

    y_train = df_train['target'].values
    y_test = df_test['target'].values

    # Store the scaler statistics and numeric column names for later use
    scaler = {
        'mean': mean.to_dict(),
        'std': std.to_dict(),
        'num_cols': num_cols,
    }
    # Invert the species mapping for returning human‑readable predictions
    species_mapping_inv = {idx: name for name, idx in species_mapping.items()}

    return X_train, X_test, y_train, y_test, dv, scaler, species_mapping_inv