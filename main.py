import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Legge direttamente il file csv, se non lo trova non va avanti
df = pd.read_csv("DataSet/super_heroes.csv")

# Crea la lista dei valori di popolarità tra i 20 eroi
#valori_popolarita = [8, 10, 7, 6, 9, 7, 10, 8, 6, 5, 6, 9, 7, 7, 9, 6, 7, 8, 7, 8]
# Aggiunta nuova colonna al DataFrame
#df['popularity'] = valori_popolarita
# File aggiornato
# df.to_csv("DataSet/super_heroes.csv", index=False)
# print("Colonna 'popularity' aggiunta!")

# Il Motore di Inferenza (Le regole del Sistema Esperto)
def sistema_esperto(row):
    # Regola per i Leader: alta intelligenza, alta popolarità
    if row['intelligence'] >= 9  and row['popularity'] >= 8:
        return 'Leader'
    # Regola per i Powerhouse: massima forza
    elif row['strength'] == 10:
        return 'Powerhouse'
    # Regola per le Powerhouse: poteri sovrannaturali, intelligenza non molto alta
    elif row['power_source'] == 'Supernatural' and row['intelligence'] < 8:
        return 'Powerhouse'
    elif row['speed'] >= 7 and row['popularity'] < 8 and row['strength'] < 8:
        return 'Specialist'
    # Regola per gli Specialist: valori bilanciati
    else:
        return 'Specialist'

# Applichiamo le regole per creare una nuova colonna 'predizione', in cui ad ogni supereroe viene assegnato il ruolo
df['predizione'] = df.apply(sistema_esperto, axis=1) # lavora per ogni singola riga axis

# Calcolo dell'Accuratezza per confrontare la predizione con i ruoli reali
acc = accuracy_score(df['role'], df['predizione'])
print(f"\nAccuratezza del sistema sui 20 eroi: {acc * 100:.2f}%") #percentuale di risposte corrette

# Generazione della Matrice di Confusione
classi_reali = df['role']
classi_predette = df['predizione']
labels = ['Leader', 'Powerhouse', 'Specialist'] # Ordine delle tre categorie

# Tabella
cm = confusion_matrix(classi_reali, classi_predette, labels=labels)

# Plottiamo la Matrice con Seaborn per renderla leggibile
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Matrice di Confusione - Validazione 20 Eroi')
plt.ylabel('Ruolo Reale')
plt.xlabel('Ruolo Predetto')
plt.show()