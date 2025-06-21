import os
import pandas as pd
import numpy as np
import joblib
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer


# ==== PATH UTILITIES ====
def get_project_path(*paths):
    """
    Build absolute path relative to this script's directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, *paths)


# ==== DATA LOADING ====
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] Dataset not found at: {file_path}")
    print(f"[INFO] Loaded dataset from {file_path}")
    return pd.read_csv(file_path)


# ==== MISSING VALUE HANDLING ====
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using saved iterative imputer and mode filling.
    """
    numerical_features = [
    'Time_spent_Alone',
    'Social_event_attendance',
    'Going_outside',
    'Friends_circle_size',
    'Post_frequency'
    ]

    categorical_features = ['Stage_fear', 'Drained_after_socializing', "Personality"]

    df_filled = df.copy()

    # Isi fitur numerik dengan median
    for col in numerical_features:
        median = df[col].median()
        df_filled[col] = df[col].fillna(median)

    for col in categorical_features:
        mode = df[col].mode()[0]
        df_filled[col] = df[col].fillna(mode)

    print(f"[INFO] Missing values handled using imputer and mode filling.")
    return df_filled


# ==== PREPROCESSING ====
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline: missing value handling, encoding, PCA.
    """
    df_filled = handle_missing_values(df)
    df_encoded = df_filled.copy()

    # Load encoders and PCA model
    encoders_path = get_project_path("models", "labelencoder.pkl")
    encoders: dict = joblib.load(encoders_path)

    pca_path = get_project_path("models", "pca.pkl")
    pca = joblib.load(pca_path)

    # Encode categorical features
    categorical_features = ['Stage_fear', 'Drained_after_socializing', "Personality"]
    
    for col in categorical_features:
        encoder = encoders[col]
        unseen_labels = set(df_encoded[col]) - set(encoder.classes_)
        if unseen_labels:
            raise ValueError(f"[ERROR] Unseen labels in column '{col}': {unseen_labels}")
        df_encoded[col] = encoder.transform(df_encoded[col])

    # Apply PCA to numerical features (ensure same ones used during PCA fit)
    pca_input_features = [
        'Social_event_attendance',
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]
    pca_features = pca.transform(df_encoded[pca_input_features])
    pca_df = pd.DataFrame(
        pca_features,
        columns=[f'PC{i+1}' for i in range(pca_features.shape[1])],
        index=df_encoded.index
    )
    
    df_encoded['Behavioral_Index'] = pca_features[:, 0]

    # Combine encoded categorical + PCA features
    # Combine encoded categorical features, PCA components, and target variable
    df_final = pd.concat([
        df_encoded['Behavioral_Index'],
        df_encoded[categorical_features],  # Encoded categorical features                           # PCA components       # Target variable
    ], axis=1)
    
    df_filled.head()
    
    print(f"[INFO] Data preprocessing complete.")
    return df_final


# ==== MAIN ENTRYPOINT ====
def main():
    input_path = get_project_path("..", "personality_dataset", "personality_dataset.csv")
    output_path = get_project_path("..", "personality_preprocessing", "personality_preprocessing.csv")

    # Load & preprocess
    df = load_dataset(input_path)
    df_preprocessed = preprocess_data(df)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_preprocessed.to_csv(output_path, index=False)
    print(f"[✓] Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    main()