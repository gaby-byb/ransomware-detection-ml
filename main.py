
from scipy.stats import randint as sp_randint
import pandas as pd
from preprocessing import preprocess  
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier


#load dataset
df = pd.read_csv("ransomware.csv")

# -- Preprocess -- 
df_clean, meta = preprocess(df)

print(df_clean.columns)

# -- split train-test data --
X = df_clean.drop(columns=['Benign']) #train
y = df_clean['Benign'] #target

#stratify handles class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

""" --Hyperparameter Tuning--
* n_estimators: the number of trees in the forest. Increasing it improves performance but increases cost
* max_depth: max depth of each tree. Reduces overfitting
"""
param_dist = {'n_estimators': sp_randint(50,500),
              'max_depth': sp_randint(1,20)}

#instance of Random forest model 
rf = RandomForestClassifier(random_state=42)

#Randomized search will sample random integers in the range given for the best  parameters 
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,               # number of random combinations to try
    cv=5,                    # 5-fold cross-validation
    scoring='f1',            # balances precision and recall
    verbose=2,
    random_state=42,
    n_jobs=-1
)
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)
print(rf_random.best_score_)