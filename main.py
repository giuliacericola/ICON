import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

# Carica DataSet
df = pd.read_csv('DataSet/super_heroes.csv')

# Converte l'attributo power_source in un valore numerico
df['power_source_encoded'] = df['power_source'].astype('category').cat.codes

# Definite le variabili X (ruolo predetto), variabili y (ruolo effettivo)
caratteristiche = ['strength', 'intelligence', 'speed', 'popularity', 'power_source_encoded']
X = df[caratteristiche]
y = df['role']

# Decision Tree
albero_esperto = DecisionTreeClassifier(
    max_depth=3,
    criterion='gini',
    random_state=10
)

# L'albero studia le relazioni numeriche nel file e genera le regole
albero_esperto.fit(X, y)

# Genera predizioni sul dataset
df['predizione_albero'] = albero_esperto.predict(X)

# Valuta l'accuratezza delle metriche
accuratezza = accuracy_score(df['role'], df['predizione_albero'])

print("=" * 60)
print(f" ACCURATEZZA DEL MODELLO AUTOMATICO: {accuratezza * 100:.2f}%")
print("=" * 60)

print("\n CLASSIFICAZIONE ")
print(classification_report(df['role'], df['predizione_albero']))

# Configurazione del k-fold: divide gli 80 eroi in 5 blocchi da 16 personaggi
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Accuratezza per ciascuna delle 5 simulazioni
print("=" * 60)
punteggi_cv = cross_val_score(albero_esperto, X, y, cv=kf, scoring='accuracy')
print("\n" " RISULTATI CROSS-VALIDATION " )
for i, score in enumerate(punteggi_cv, 1):
    print(f" Simulazione {i} (su 16 eroi): {score * 100:.2f}%")
print("\n")
print(f" Accuratezza MEDIA Reale: {punteggi_cv.mean() * 100:.2f}%")
print(f" Deviazione Standard : +/- {punteggi_cv.std() * 100:.2f}%")
print("=" * 60)


# Rappresentazione Visiva
regole_strutturate = export_text(
    albero_esperto,
    feature_names=caratteristiche
)
print("\n" + "=" * 20 + " Struttura Regole " + "=" * 20)
print(regole_strutturate)
print("=" * 67)

# Controllo degli errori
errori = df[df['role'] != df['predizione_albero']]
if not errori.empty:
    print(f"\n PERSONAGGI FUORI POSTO ({len(errori)})")
    for index, row in errori.iterrows():
        print(f"Eroe: {row['name']} | Ruolo del DataSet: {row['role']} | Ruolo predetto: {row['predizione_albero']}")
else:
    print("\n Accuratezza al 100%! ")