import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Caricamento del file csv
def load_dataset(path):
    return pd.read_csv(path)

# Converte le stringhe in valori numerici
# Parametro da passare come input della libreria pandas per inizializzare un DataSet
# Input codificato correttamente aventi tutti gli attributi in valore numerico
# Se l'ontologia ha inserito i flag booleani, li aggiungiamo alle caratteristiche
# X definisce le caratteristiche e y il ruolo
def preprocessing_dataset(df_input):
    df_input['power_source_encoded'] = df_input['power_source'].astype('category').cat.codes
    characteristic = ['strength', 'intelligence', 'speed', 'popularity', 'power_source_encoded']

    if 'is_cosmic' in df_input.columns:
        characteristic.append('is_cosmic')
    if 'is_glass_cannon' in df_input.columns:
        characteristic.append('is_glass_cannon')

    X = df_input[characteristic]
    y = df_input['role']
    return X, y

# Funzione che addestra l'albero di decisione, valuta le metriche
# Definita: profondità pari a 3, metrica utilizzata gini, generatore di numeri casuali pari a 10
# Genera predizioni sul dataset, utilizza una copia in locale
# Il sistema valuta l'accuratezza di tale modello
# Visualizzazione della Classificazione (precision, recall, f1-score, support, accuracy, macro avg, weighted avg)
# Visualizzazione delle regole che ha applicato il sistema esperto
# La funzione inoltre mostra gli errori

def decisiontree_classifier(X, y, df_originale):
    decision_tree = DecisionTreeClassifier(
        max_depth=3,
        criterion='gini',
        random_state=10
    )

    decision_tree.fit(X, y)

    df_pred = df_originale.copy()
    df_pred['predizione_albero'] = decision_tree.predict(X)

    accurate = accuracy_score(df_pred['role'], df_pred['predizione_albero'])

    print("=" * 60)
    print(f" ACCURATEZZA DEL MODELLO AUTOMATICO: {accurate * 100:.2f}%")
    print("=" * 60)

    print("\n CLASSIFICAZIONE ")
    print(classification_report(df_pred['role'], df_pred['predizione_albero']))

    characteristic = list(X.columns)
    rules = export_text(decision_tree, feature_names=characteristic)
    print("\n" + "=" * 20 + " Struttura Regole " + "=" * 20)
    print(rules)
    print("=" * 67)

    errors = df_pred[df_pred['role'] != df_pred['predizione_albero']]
    if not errors.empty:
        print(f"\n PERSONAGGI FUORI POSTO ({len(errors)})")
        for index, row in errors.iterrows():
            print(
                f"Eroe: {row['name']} | Ruolo del DataSet: {row['role']} | Ruolo predetto: {row['predizione_albero']}")
    else:
        print("\n Accuratezza al 100%! ")

    return decision_tree

# Funzione che esegue la Cross-Validation e genera il grafico della curva di validazione.
# Configurazione del k-fold: divide i 130 eroi in 5 blocchi da 26 personaggi
# Albero di riferimento a profondità 3 per la stampa della cross-validation
# Visualizzazione dei risultati ottenuti dalle varie simulazioni
# Visualizzazione del grafico, nel quale vi sono due curve:
# 1) Accuratezza del DataSet 2) Accuratezza Media della Cross Validation

def run_cross_validation_and_plots(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    decision_tree = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=10)

    print("=" * 60)
    score_cv = cross_val_score(decision_tree, X, y, cv=kf, scoring='accuracy')
    print("\n RISULTATI CROSS-VALIDATION ")
    for i, score in enumerate(score_cv, 1):
        print(f" Simulazione {i} (su 26 eroi): {score * 100:.2f}%")
    print("\n")
    print(f" Accuratezza MEDIA Reale: {score_cv.mean() * 100:.2f}%")
    print(f" Deviazione Standard : +/- {score_cv.std() * 100:.2f}%")
    print("=" * 60)

    depth_to_test = [1, 2, 3, 4, 5]
    acc_complete_dataset = []
    acc_cross_validation = []

    for depth in depth_to_test:
        tree_temp = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=10)

        # Accuratezza sul Data Set completo
        tree_temp.fit(X, y)
        pred_temp = tree_temp.predict(X)
        acc_comp = np.mean(pred_temp == y)
        acc_complete_dataset.append(acc_comp * 100)

        # Accuratezza Cross-Validation
        score_cv_temp = cross_val_score(tree_temp, X, y, cv=kf, scoring='accuracy')
        acc_cross_validation.append(score_cv_temp.mean() * 100)

    # Disegno del grafico visivo
    plt.figure(figsize=(10, 6))

    # Linea del Dataset Completo (Training)
    plt.plot(depth_to_test, acc_complete_dataset, marker='o', linewidth=2, color='green',
             label='DataSet (Training)')

    # Linea della Cross-Validation (Test)
    plt.plot(depth_to_test, acc_cross_validation, marker='s', linewidth=2, color='orange',
             label='Cross-Validation (Test)')

    # Evidenziazione visiva del punto ottimale impostato a profondità 3
    plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Scelta Corrente (Profondità = 3)')

    plt.title("Curva di Validazione: Dataset vs Cross-Validation", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Profondità Massima dell'Albero (max_depth)", fontsize=12)
    plt.ylabel("Accuratezza (%)", fontsize=12)
    plt.xticks(depth_to_test)
    plt.ylim(50, 102)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')

    # Apertura finestra del grafico
    plt.show()