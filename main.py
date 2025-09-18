
#Import dataset from Kaggle

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("ransomware.csv")

#Preprocessing

    ## Machine

#drop invalid options
df = df[~df['Machine'].isin([0, 870])]
#map raw codes to string labels
machine_map = {
    332: 'x86',
    34404: 'x64',
    452: 'ARM',
    43620: 'ARM64'
}
#Map int to strings
df['Machine'] = df['Machine'].map(machine_map)
#One-Hot Encode
df = pd.get_dummies(df, columns=['Machine'], prefix='Machine')

print(df.columns)