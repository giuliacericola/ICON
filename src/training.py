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
# X definisce le caratteristiche e y il ruolo
def preprocessing_dataset(df_input):
    df_input['power_source_encoded'] = df_input['power_source'].astype('category').cat.codes
    characteristic = ['strength', 'intelligence', 'speed', 'popularity', 'power_source_encoded']

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

def decisiontree_classifier(X, y, df_original):
    decision_tree = DecisionTreeClassifier(
        max_depth=3,
        criterion='gini',
        random_state=10
    )

    decision_tree.fit(X, y)

    df_pred = df_original.copy()
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

    score_cv_pct = score_cv * 100
    sim = [
        'Sim. 1\n(26 eroi)',
        'Sim. 2\n(26 eroi)',
        'Sim. 3\n(26 eroi)',
        'Sim. 4\n(26 eroi)',
        'Sim. 5\n(26 eroi)'
    ]

    plt.figure(figsize=(10, 6))

    # Linea della CV
    plt.plot(sim, score_cv_pct, marker='s', linewidth=2, color='orange',
             label='Cross-Validation (Test Fold)')

    # Linea della media reale della Cross-Validation
    plt.axhline(y=score_cv.mean() * 100, color='red', linestyle='--', alpha=0.7,
                label=f'Accuratezza Media Reale ({score_cv.mean() * 100:.2f}%)')

    # Linea dell'accuratezza fissa dell'albero sul Dataset
    plt.axhline(y=acc_complete_dataset[2], color='green', linestyle=':', alpha=0.7,
                label=f'DataSet Completo (Training Base: {acc_complete_dataset[2]:.2f}%)')

    plt.title("Comportamento della Cross-Validation nelle 5 Simulazioni di Test", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Numero di Eroi", fontsize=12)
    plt.ylabel("Accuratezza (%)", fontsize=12)
    plt.ylim(50, 102)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    plt.show()

# Funzione che esegue una 5-Fold Cross-Validation simulando il comportamento della pipeline ibrida
# Ad ogni fold l'albero predice e l'ontologia interviene a correggere gli errori
# Genera un grafico che mette in contrapposizione ML e ML+Ontologia
def run_hybrid_cross_validation_and_plot(X, y, df_full):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    score_ml = []
    score_mlo = []

    print("\n" + "=" * 60)
    print(" AVVIO CROSS-VALIDATION SULLA PIPELINE IBRIDA (ML + ONTOLOGIA) ")
    print("=" * 60)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        tree_fold = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=10)
        tree_fold.fit(X_train, y_train)

        pred_tree_test = tree_fold.predict(X_test)
        acc_single_tree = np.mean(pred_tree_test == y_test)
        score_ml.append(acc_single_tree * 100)

        # Filtro di intervento a posteriori dell'ontologia sugli errori di questo fold
        ibrid = 0
        for i, idx in enumerate(test_idx):
            real_role = y.iloc[idx]
            predict_role = pred_tree_test[i]
            ruolo_ontologia = df_full.loc[
                idx, 'ruolo_ontologia'] if 'ruolo_ontologia' in df_full.columns else None

            if predict_role != real_role:
                # Se l'albero sbaglia, verifichiamo se l'ontologia ha la risposta corretta
                if pd.notna(ruolo_ontologia) and ruolo_ontologia == real_role:
                    ibrid += 1
            else:
                ibrid += 1

        acc_single_ibrid = ibrid / len(test_idx)
        score_mlo.append(acc_single_ibrid * 100)

        print(f" Simulazione {fold} (su {len(test_idx)} eroi):")
        print(f"   -> Accuratezza Decision Tree:     {acc_single_tree * 100:.2f}%")
        print(f"   -> Accuratezza Decision Tree + Ontologia:   {acc_single_ibrid * 100:.2f}%")
        print("-" * 50)

    print("\n RISULTATI MEDI CONFRONTATI IN Cross-Validation")
    print(f" Accuratezza MEDIA Solo ML:        {np.mean(score_ml):.2f}%")
    print(f" Accuratezza MEDIA ML + Ontologia: {np.mean(score_mlo):.2f}%")
    print("=" * 60)

    # Disegno del 3° Grafico: Confronto della Cross-Validation Ibrida
    sim_heroes = ['Sim. 1\n(26 eroi)', 'Sim. 2\n(26 eroi)', 'Sim. 3\n(26 eroi)', 'Sim. 4\n(26 eroi)',
                        'Sim. 5\n(26 eroi)']

    plt.figure(figsize=(10, 6))

    # Curva della Cross-Validation Ml
    plt.plot(sim_heroes, score_ml, marker='s', linewidth=2, color='orange',
             label='Cross-Validation Classica (Solo ML)')

    # Curva della Cross-Validation Ml+Ontologia
    plt.plot(sim_heroes, score_mlo, marker='^', linewidth=2, color='red',
             label='Cross-Validation Ibrida (ML + Ontologia)')

    plt.title("Andamento della Cross-Validation: Modello ML vs Modello Ibrido Semantico", fontsize=13,
              fontweight='bold', pad=15)
    plt.xlabel("Simulazioni di Test (Numero di Eroi)", fontsize=12)
    plt.ylabel("Accuratezza (%)", fontsize=12)
    plt.ylim(50, 105)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right')

    # Stampa i valori numerici
    for i in range(len(sim_heroes)):
        plt.text(sim_heroes[i], score_ml[i] - 3.5, f'{score_ml[i]:.1f}%', ha='center',
                 color='orange', fontweight='bold')
        plt.text(sim_heroes[i], score_mlo[i] + 1.5, f'{score_mlo[i]:.1f}%',
                 ha='center', color='red', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Genera un grafico a baree che confronta il numero di eroi indovinati dal Decision Tree e
# il numero di eroi indovinati dal Decision Tree con l'aggiunta dell'ontologia
def plot_heroes_comparison(heroes_dt_correct, heroes_hybrid_correct, total_heroes):

    category = ['Eroi Indovinati\nda Decision Tree (ML)', 'Eroi Indovinati\nda ML + Ontologia']
    values = [heroes_dt_correct, heroes_hybrid_correct]
    colors = ['#ff7675', '#55efc4']

    plt.figure(figsize=(8, 6))

    # Creazione delle barre (Corretto: rimosso l'argomento errato 'values=')
    bars = plt.bar(category, values, color=colors, width=0.4, edgecolor='black', linewidth=1.2)

    # Configurazione assi e titolo
    plt.title("Impatto dell'Ontologia sul Numero di Eroi Classificati Correttamente", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Numero di Eroi (Conteggio Assoluto)", fontsize=12)

    # Limite massimo dell'ordinata pari al totale degli eroi + un piccolo margine per il testo
    plt.ylim(0, total_heroes + 15)
    plt.grid(axis='y', linestyle=':', alpha=0.6)

    # Inserisce il numero esatto sopra ogni bar
    for b in bars:
        height = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2.0, height + 2,
                 f'{int(height)} / {total_heroes}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()