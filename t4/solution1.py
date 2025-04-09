## Exercise 1 (10 minutes): Load & Preprocess Your Dataset
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Încărcăm setul de date Iris din scikit-learn
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Introducem valori lipsă artificiale (opțional, pentru demonstrare)
#    Vom seta câteva valori NaN în coloana 'petal length (cm)'
df.iloc[5:10, 2] = np.nan

# 3. Tratarea valorilor lipsă
#    Vom folosi SimpleImputer pentru a înlocui NaN-urile cu media fiecărei coloane
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 4. Scalarea datelor
#    Folosim StandardScaler pentru a transforma fiecare caracteristică astfel încât media să fie 0 și abaterea standard să fie 1
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df.columns)

# 5. Verificarea rezultatelor
print(f"Dimensiunea setului de date: {df_scaled.shape}")
print("\nStatistici de bază:")
print(df_scaled.describe())

# 6. (Opțional) Afișarea primelor câteva rânduri pentru a confirma preprocesarea
print("\nPrimele 5 rânduri ale setului de date preprocesat:")
print(df_scaled.head())
