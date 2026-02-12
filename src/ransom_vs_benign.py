import warnings
import joblib
warnings.filterwarnings("ignore")
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from evaluate import evaluate_model

from xgboost import XGBClassifier


# --- Load Dataset
df = pd.read_csv("data/ransom_vs_benign.csv")

# --- Preprocess ---
# function returns target and train data
def preprocess(df):


    df = df[~df['Machine'].isin([0, 870])].copy()  # drop invalid options
        
    #map raw codes to string labels
    machine_map = {
            332: 'x86',
            34404: 'x64',
            452: 'ARM',
            43620: 'ARM64'
    }
    #map string to 0-1 labels
    df['Machine'] = df['Machine'].map(machine_map)
    df = pd.get_dummies(df, columns=['Machine'], prefix='Machine')
        
    # Get rid of Identifiers ---
    meta = df[['FileName', 'md5Hash']].copy()
    df = df.drop(columns=['FileName', 'md5Hash'])

    #Create a new target column Ransom
    df["Ransomware"] = df["Benign"].map({1:0, 0:1})
    df = df.drop(columns=["Benign"])

    X = df.drop(columns=['Ransomware']) #train
    y = df['Ransomware'] #target

    return X, y, meta


X, y, meta = preprocess(df)


# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)


#----- RANDOM FOREST -----#
#best parameters found using RandomSearch (see ipynb)

rf = RandomForestClassifier(max_depth= 30, 
                            min_samples_leaf= 1, 
                            min_samples_split= 3, 
                            n_estimators= 252,
                            random_state=42)

# Train
rf.fit(X_train, y_train)


#----- XGBoost -----#

#best parameters found with gridSearch (see ipynb)
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train),  # balance weight
    random_state=42,
    eval_metric='logloss'  # suppresses warning
)


xgb.fit(X_train, y_train)



# ---- SVM ----

#Only need to scale data for SVM model 
# Best params based on ipynb search 
final_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        C=10,
        dual=False,
        max_iter=20000,
        tol=1e-3,
        class_weight="balanced"
    ))
])

final_svm.fit(X_train, y_train)

# Evaluation:

# Random Forest
evaluate_model(rf, X_test, y_test, model_name="Random Forest")

# XGBoost
evaluate_model(xgb, X_test, y_test, model_name="XGBoost")

# SVM 
evaluate_model(final_svm, X_test, y_test, model_name="SVM")

# --- for the  CLI ----

joblib.dump(rf, "models/ransomware_detector.pkl")

#save dataset without labels to test
X_train.to_csv("data/X_ransom_vs_benign.csv", index=False)

#save labels for confirmation
y_train.to_csv("data/labels_ransom_vs_benign.csv", index=False)
