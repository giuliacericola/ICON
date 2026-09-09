import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Caricamento del file csv
def load_dataset(path):
    return pd.read_csv(path)

# Metodo che converte le stringhe in valori numerici
# Parametro da passare come input della libreria pandas per inizializzare un DataSet
# Input codificato correttamente aventi tutti gli attributi in valore numerico
# X definisce le caratteristiche e y il ruolo
def preprocessing_dataset(df_input):
    df_input['power_source_encoded'] = df_input['power_source'].astype('category').cat.codes
    characteristic = ['strength', 'intelligence', 'speed', 'power_source_encoded']

    X = df_input[characteristic]
    y = df_input['role']
    return X, y

# Funzione che addestra l'albero di decisione sull'intero dataset ed estrae le
# regole decisionali leggibili (interpretabilita' del modello simbolico).
# Definita: profondita' pari a 3, metrica utilizzata gini, generatore di numeri casuali pari a 10.
# L'accuratezza calcolata qui e' in-sample (l'albero viene testato
# sugli stessi dati con cui e' stato addestrato) e NON costituisce la metrica
# di valutazione ufficiale del progetto. Serve solo per due scopi diagnostici.
# La valutazione ufficiale delle prestazioni del modello e' condotta
# esclusivamente tramite K-Fold Cross-Validation che riportano
# media e deviazione standard su piu' fold.
def decisiontree_classifier(X, y, df_original):
    decision_tree = DecisionTreeClassifier(
        max_depth=3,
        criterion='gini',
        random_state=10
    )

    decision_tree.fit(X, y)

    df_pred = df_original.copy()
    df_pred['predizione_albero'] = decision_tree.predict(X)

    accurate_in_sample = accuracy_score(df_pred['role'], df_pred['predizione_albero'])

    print("=" * 60)
    print(" FIT SU TRAINING SET COMPLETO (solo a scopo diagnostico/interpretativo)")
    print(" Le metriche di valutazione ufficiali sono riportate piu' avanti")
    print(" tramite K-Fold Cross-Validation (media e deviazione standard).")
    print("=" * 60)
    print(f" Accuratezza in-sample (non valutativa): {accurate_in_sample * 100:.2f}%")

    characteristic = list(X.columns)
    rules = export_text(decision_tree, feature_names=characteristic)
    print("\n" + "=" * 20 + " Struttura Regole Apprese dall'Albero " + "=" * 20)
    print(rules)
    print("=" * 67)

    errors = df_pred[df_pred['role'] != df_pred['predizione_albero']]
    if not errors.empty:
        print(f"\n PERSONAGGI CHE VIOLANO ANCHE IL PATTERN DI TRAINING ({len(errors)})")
        print(" (utile come indicatore di outlier nel dataset, non come metrica)")
        for index, row in errors.iterrows():
            print(
                f"Eroe: {row['name']} | Ruolo del DataSet: {row['role']} | Ruolo predetto: {row['predizione_albero']}")
    else:
        print("\n Nessun personaggio viola il pattern appreso sul training set.")

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
    print("\n RISULTATI CROSS-VALIDATION (metrica ufficiale del progetto)")
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

    print("Generazione grafico sulla scelta ottimale della profondità dell'albero...")
    consent = input(">>Vuoi visualizzare il grafico? (s/n): ").strip().lower()
    if consent == 's':
        plt.figure(figsize=(10, 6))
        plt.plot(depth_to_test, acc_complete_dataset, marker='o', linewidth=2, color='green',
                 label='DataSet (Training)')
        plt.plot(depth_to_test, acc_cross_validation, marker='s', linewidth=2, color='orange',
                 label='Cross-Validation (Test)')
        plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Scelta Corrente (Profondità = 3)')
        plt.title("Curva di Validazione: Dataset vs Cross-Validation", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Profondità Massima dell'Albero (max_depth)", fontsize=12)
        plt.ylabel("Accuratezza (%)", fontsize=12)
        plt.xticks(depth_to_test)
        plt.ylim(50, 102)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11, loc='lower right')
        plt.show()
    else: print("Visualizzazione grafico Annullata, il programma prosegue...")

    score_cv_pct = score_cv * 100
    sim = [
        'Simulazione 1\n(26 eroi)',
        'Simulazione 2\n(26 eroi)',
        'Simulazione 3\n(26 eroi)',
        'Simulazione 4\n(26 eroi)',
        'Simulazione 5\n(26 eroi)'
    ]

    print("Generazione Grafico sull'accuratezza della Cross-Validation ")
    consent = input(">>Vuoi visualizzare il grafico? (s/n): ").strip().lower()
    if consent == 's':
        plt.figure(figsize=(10, 6))
        plt.plot(sim, score_cv_pct, marker='s', linewidth=2, color='orange',
                 label='Cross-Validation (Test Fold)')
        plt.axhline(y=score_cv.mean() * 100, color='red', linestyle='--', alpha=0.7,
                    label=f'Accuratezza Media Reale ({score_cv.mean() * 100:.2f}%)')
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
    else: print("Visualizzazione Grafico Annullata, il programma prosegue...")

# Restituisce i due modelli di apprendimento supervisionato da confrontare
# comparativamente: Decision Tree e KNN.
def build_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=10),
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    }

# Metodo che esegue la stessa procedura di scelta del parametro usata per il Decision Tree
# (curva di validazione in CV), qui applicata al numero di vicini k del k-NN.
# Necessaria la standardizzazione (StandardScaler) perché il k-NN si basa
# sulla distanza euclidea ed è sensibile alla scala delle feature.
def run_knn_k_selection_and_plot(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_to_test = [1, 3, 5, 7, 9, 11]
    acc_cross_validation = []

    print("=" * 60)
    print(" SCELTA DEL PARAMETRO k (k-NN) TRAMITE CROSS-VALIDATION")
    print("=" * 60)

    for k in k_to_test:
        modello_temp = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        score_cv_temp = cross_val_score(modello_temp, X, y, cv=kf, scoring='accuracy')
        acc_cross_validation.append(score_cv_temp.mean() * 100)
        print(f" k={k:<3} -> Accuratezza media CV: {score_cv_temp.mean() * 100:.2f}% "
              f"(+/- {score_cv_temp.std() * 100:.2f}%)")

    print("=" * 60)

    print("Generazione grafico sulla scelta ottimale di k...")
    consent = input(">>Vuoi visualizzare il grafico? (s/n): ").strip().lower()
    if consent == 's':
        plt.figure(figsize=(10, 6))
        plt.plot(k_to_test, acc_cross_validation, marker='o', linewidth=2, color='#0984e3',
                 label='Cross-Validation (Test)')
        plt.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Scelta Corrente (k = 5)')
        plt.title("Curva di Validazione: scelta del numero di vicini k", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Numero di vicini (k)", fontsize=12)
        plt.ylabel("Accuratezza Media CV (%)", fontsize=12)
        plt.xticks(k_to_test)
        plt.ylim(50, 102)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11, loc='lower right')
        plt.show()
    else:
        print("Visualizzazione grafico Annullata, il programma prosegue...")

# Confronta Decision Tree e k-NN in 5-Fold Cross-Validation.
# Serve a stabilire, su base empirica, quale dei due modelli statistici sia
# il piu' accurato: il vincitore verra' poi usato come modello di riferimento
# per l'integrazione con la KB.
def run_solo_ml_comparison(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    modelli = build_models()

    risultati = {}

    print("\n" + "=" * 70)
    print(" CONFRONTO SOLO-ML: DECISION TREE vs k-NN")
    print(" (5-Fold Cross-Validation, nessun intervento dell'ontologia)")
    print("=" * 70)

    for nome_modello, modello in modelli.items():
        score_ml = []
        print(f"\n--- Modello: {nome_modello} ---")

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            modello_fold = clone(modello)
            modello_fold.fit(X_train, y_train)
            pred_test = modello_fold.predict(X_test)

            acc = np.mean(pred_test == y_test.values)
            score_ml.append(acc * 100)
            print(f" Fold {fold}: {acc * 100:.2f}%")

        risultati[nome_modello] = {
            'media': float(np.mean(score_ml)),
            'std': float(np.std(score_ml))
        }

    print("\n" + "=" * 70)
    print(" TABELLA RIASSUNTIVA SOLO-ML (media +/- deviazione standard su 5 fold)")
    print("=" * 70)
    print(f" {'Modello':<15} | {'Accuratezza CV'}")
    print("-" * 70)
    for nome_modello, r in risultati.items():
        print(f" {nome_modello:<15} | {r['media']:.2f}% (+/- {r['std']:.2f}%)")
    print("=" * 70)

    return risultati

# Grafico a barre del solo confronto Decision Tree vs k-NN.
def plot_solo_ml_comparison(risultati):

    print("Generazione grafico comparativo Solo-ML (Decision Tree vs k-NN)...")
    consent = input(">>Vuoi visualizzare il grafico? (s/n): ").strip().lower()

    if consent == 's':
        modelli_nomi = list(risultati.keys())
        medie = [risultati[m]['media'] for m in modelli_nomi]
        std = [risultati[m]['std'] for m in modelli_nomi]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(modelli_nomi, medie, yerr=std, capsize=6,
                        color=['#0984e3', '#fdcb6e'], width=0.4, edgecolor='black', linewidth=1.0)
        plt.ylabel("Accuratezza Media CV (%)", fontsize=12)
        plt.title("Confronto Solo-ML: Decision Tree vs k-NN\n(media e deviazione standard su 5-Fold CV)",
                  fontsize=13, fontweight='bold', pad=15)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        for b, m, s in zip(bars, medie, std):
            plt.text(b.get_x() + b.get_width() / 2, m + s + 1.5, f'{m:.2f}%',
                      ha='center', fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print("Visualizzazione Grafico Annullata, il programma prosegue...")

# Funzione che valuta la pipeline ibrida (ML + Ontologia) in 5-Fold Cross-Validation
# per il solo Decision Tree, che nella fase Solo-ML è risultato il modello con
# l'accuratezza piu' alta e viene perciò scelto come modello di riferimento
# per l'integrazione con la conoscenza simbolica.
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

        ibrid = 0
        for i, idx in enumerate(test_idx):
            real_role = y.iloc[idx]
            predict_role = pred_tree_test[i]
            ruolo_ontologia = df_full.loc[
                idx, 'ruolo_ontologia'] if 'ruolo_ontologia' in df_full.columns else None

            if predict_role != real_role:
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
    print(f" Deviazione Standard Solo ML:      +/- {np.std(score_ml):.2f}%")
    print(f" Accuratezza MEDIA ML + Ontologia: {np.mean(score_mlo):.2f}%")
    print(f" Deviazione Standard ML + Ontologia: +/- {np.std(score_mlo):.2f}%")
    print("=" * 60)

    print("Generazione Grafico: confronto delle curve di Cross_Validation ML vs ML+Ontologia")
    consent = input(">>Vuoi visualizzare il grafico? (s/n): ").strip().lower()
    if consent == 's':
        sim_heroes = ['Sim. 1\n(26 eroi)', 'Sim. 2\n(26 eroi)', 'Sim. 3\n(26 eroi)', 'Sim. 4\n(26 eroi)',
                            'Sim. 5\n(26 eroi)']

        plt.figure(figsize=(10, 6))
        plt.plot(sim_heroes, score_ml, marker='s', linewidth=2, color='orange',
                 label='Cross-Validation Classica (Solo ML)')
        plt.plot(sim_heroes, score_mlo, marker='^', linewidth=2, color='red',
                 label='Cross-Validation Ibrida (ML + Ontologia)')
        plt.title("Andamento della Cross-Validation: Modello ML vs Modello Ibrido Semantico", fontsize=13,
                  fontweight='bold', pad=15)
        plt.xlabel("Simulazioni di Test (Numero di Eroi)", fontsize=12)
        plt.ylabel("Accuratezza (%)", fontsize=12)
        plt.ylim(50, 105)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11, loc='lower right')

        for i in range(len(sim_heroes)):
            plt.text(sim_heroes[i], score_ml[i] - 3.5, f'{score_ml[i]:.1f}%', ha='center',
                     color='orange', fontweight='bold')
            plt.text(sim_heroes[i], score_mlo[i] + 1.5, f'{score_mlo[i]:.1f}%',
                     ha='center', color='red', fontweight='bold')

        plt.tight_layout()
        plt.show()
    else: print("Visualizzazione Grafico Annullata, il programma prosegue...")

# Verifica di generalita': l'intervento dell'ontologia sui fallimenti del ML
# viene ora applicato anche al k-NN, per accertare che il miglioramento
# osservato con il Decision Tree non dipenda dal particolare tipo di errori
# commessi da un singolo classificatore, ma sia una correzione semantica che
# generalizza tra modelli statistici diversi. Per ogni modello e ogni
# configurazione vengono riportate media e deviazione standard su 5 fold.
def run_hybrid_multi_model_comparison(X, y, df_full):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    modelli = build_models()

    risultati = {}

    print("\n" + "=" * 70)
    print(" VERIFICA DI GENERALITA': L'INTERVENTO DELL'ONTOLOGIA SI ESTENDE AD ALTRI MODELLI?")
    print(" (5-Fold Cross-Validation, con e senza intervento dell'ontologia)")
    print("=" * 70)

    for nome_modello, modello in modelli.items():
        score_ml = []
        score_ibrido = []

        print(f"\n--- Modello: {nome_modello} ---")

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            modello_fold = clone(modello)
            modello_fold.fit(X_train, y_train)
            pred_test = modello_fold.predict(X_test)

            acc_solo_ml = np.mean(pred_test == y_test.values)
            score_ml.append(acc_solo_ml * 100)

            ibrid = 0
            for i, idx in enumerate(test_idx):
                real_role = y.iloc[idx]
                predict_role = pred_test[i]
                ruolo_ontologia = df_full.loc[
                    idx, 'ruolo_ontologia'] if 'ruolo_ontologia' in df_full.columns else None

                if predict_role != real_role:
                    if pd.notna(ruolo_ontologia) and ruolo_ontologia == real_role:
                        ibrid += 1
                else:
                    ibrid += 1

            acc_ibrido = ibrid / len(test_idx)
            score_ibrido.append(acc_ibrido * 100)

            print(f" Fold {fold}: Solo ML = {acc_solo_ml * 100:.2f}% | ML+Ontologia = {acc_ibrido * 100:.2f}%")

        risultati[nome_modello] = {
            'ml_media': float(np.mean(score_ml)),
            'ml_std': float(np.std(score_ml)),
            'ibrido_media': float(np.mean(score_ibrido)),
            'ibrido_std': float(np.std(score_ibrido))
        }

    print("\n" + "=" * 70)
    print(" TABELLA RIASSUNTIVA (media +/- deviazione standard su 5 fold)")
    print("=" * 70)
    print(f" {'Modello':<15} | {'Solo ML':<24} | {'ML + Ontologia':<24}")
    print("-" * 70)
    for nome_modello, r in risultati.items():
        colonna_ml = f"{r['ml_media']:.2f}% (+/- {r['ml_std']:.2f}%)"
        colonna_ibrido = f"{r['ibrido_media']:.2f}% (+/- {r['ibrido_std']:.2f}%)"
        print(f" {nome_modello:<15} | {colonna_ml:<24} | {colonna_ibrido:<24}")
    print("=" * 70)

    return risultati

# Genera un grafico a barre raggruppate che confronta i due modelli
# (Decision Tree e k-NN), sia in versione solo-ML sia ibrida con l'ontologia,
# con le barre di errore rappresentanti la deviazione standard su 5 fold.
def plot_model_comparison(risultati):
    print("Generazione grafico comparativo tra i modelli (Solo ML vs ML+Ontologia)...")
    consent = input(">>Vuoi visualizzare il grafico? (s/n): ").strip().lower()
    if consent == 's':
        modelli_nomi = list(risultati.keys())
        ml_medie = [risultati[m]['ml_media'] for m in modelli_nomi]
        ml_std = [risultati[m]['ml_std'] for m in modelli_nomi]
        ibrido_medie = [risultati[m]['ibrido_media'] for m in modelli_nomi]
        ibrido_std = [risultati[m]['ibrido_std'] for m in modelli_nomi]

        x = np.arange(len(modelli_nomi))
        width = 0.35

        plt.figure(figsize=(9, 6))
        plt.bar(x - width / 2, ml_medie, width, yerr=ml_std, capsize=6, label='Solo ML', color='#ff7675')
        plt.bar(x + width / 2, ibrido_medie, width, yerr=ibrido_std, capsize=6, label='ML + Ontologia', color='#55efc4')

        plt.xticks(x, modelli_nomi)
        plt.ylabel("Accuratezza Media CV (%)", fontsize=12)
        plt.title("Confronto tra Modelli: Solo ML vs ML + Ontologia\n(media e deviazione standard su 5-Fold CV)",
                  fontsize=13, fontweight='bold', pad=15)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()
    else:
        print("Visualizzazione Grafico Annullata, il programma prosegue...")

# Genera un grafico a barre che confronta il numero di eroi indovinati dal Decision Tree e
# il numero di eroi indovinati dal Decision Tree con l'aggiunta dell'ontologia
def plot_heroes_comparison(heroes_dt_correct, heroes_hybrid_correct, total_heroes):

    print("Generazione istogrammi a confronto: ML vs Ml+Ontologia. EROI INDOVINATI...")
    consent = input(">>Vuoi visualizzare il grafico a barre? (s/n): ").strip().lower()
    if consent == 's':
        category = ['Eroi Indovinati\nda Decision Tree (ML)', 'Eroi Indovinati\nda ML + Ontologia']
        values = [heroes_dt_correct, heroes_hybrid_correct]
        colors = ['#ff7675', '#55efc4']

        plt.figure(figsize=(8, 6))

        bars = plt.bar(category, values, color=colors, width=0.4, edgecolor='black', linewidth=1.2)

        plt.title("Impatto dell'Ontologia sul Numero di Eroi Classificati Correttamente", fontsize=13, fontweight='bold', pad=15)
        plt.ylabel("Numero di Eroi (Conteggio Assoluto)", fontsize=12)

        plt.ylim(0, total_heroes + 15)
        plt.grid(axis='y', linestyle=':', alpha=0.6)

        for b in bars:
            height = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2.0, height + 2,
                     f'{int(height)} / {total_heroes}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()
    else: print("Visualizazione Grafica Annullata, il programma prosegue...")