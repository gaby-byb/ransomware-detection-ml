import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

def preprocess(df):
    """
    Preprocess ransomware dataset for different model types:
    - Clean invalid Machine codes
    - Map Machine to string labels
    - One-hot encode categorical features
    - Save identifiers separately (FileName, md5Hash)
    - Drop identifiers from training set
    - Scale features (safe to do for all models)
    """
    # --- Clean invalid Machine codes ---
    df = df[~df['Machine'].isin([0, 870])].copy()  # drop invalid options
    
    #One Hot Encode: raw codes -> string labels -> Binary
    machine_map = {
        332: 'x86',
        34404: 'x64',
        452: 'ARM',
        43620: 'ARM64'
    }
    df['Machine'] = df['Machine'].map(machine_map)
    df = pd.get_dummies(df, columns=['Machine'], prefix='Machine')
    
    # --- Identifiers ---
    meta = df[['FileName', 'md5Hash']].copy()
    df = df.drop(columns=['FileName', 'md5Hash'])
    
    # --- Split Features vs Target ---
    X = df.drop(columns=['Benign']) #Train
    y = df['Benign'] #target


    # --- Feature Scalling - Needed for SVM model ---
    scaler = StandardScaler()

    #scales each row to have mean = 0, std = 1.
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)


    
    return X, y, meta
