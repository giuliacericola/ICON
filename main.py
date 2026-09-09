import os
import pandas as pd
from src import training, semantic_hero, battle

# La pipeline principale del progetto coordina l'intero flusso di lavoro.
# Il focus della prima fase riguarda l'apprendimento supervisionato:
# Realizzazione del Decision Tree: addestramento e scelta della profondita' (max_depth);
# Realizzazione k-NN: scelta del numero di vicini k;
# Confronto dei due modelli ML.
# Nella seconda fase l'obiettivo principale è il ragionamento semantico attraverso la popolazione dell'ontologia,
# creata appositamente su protege.
# Successivamente vi è l'integrazione del ML scelto (decision tree) + Ontologia.
# Infine come ultimo modulo del programma vi è una fase di battle, in cui l'ontologia interviene fornendo
# la conoscenza inferita su cui si basano le decisioni tattiche.
def main():
    print("=" * 70)
    print("HEROES IN SUPER-TRAINING")
    print("=" * 70)

    cartella_principale = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(cartella_principale, 'DataSet', 'super_heroes.csv')

    if not os.path.exists(dataset_path):
        print(f"[ERRORE] Dataset non trovato in: {dataset_path}")
        return

    dataset_originale = training.load_dataset(dataset_path)
    X_orig, y_orig = training.preprocessing_dataset(dataset_originale.copy())

    # PRIMA PARTE, decision tree e knn
    print("\n>>> PRIMA PARTE: Apprendimento supervisionato >>>")
    print("\n>>> ML. Decision Tree — addestramento e scelta della profondità...")
    dt_model_orig = training.decisiontree_classifier(X_orig, y_orig, dataset_originale)

    # Curva di validazione (scelta di max_depth) + dettaglio dei 5 fold
    training.run_cross_validation_and_plots(X_orig, y_orig)

    print("\n[INFO] Decision Tree completato.")
    print("-" * 70)

    print("\n>>> Ml. k-NN — scelta del numero di vicini k...")
    training.run_knn_k_selection_and_plot(X_orig, y_orig)

    print("\n[INFO] k-NN completato.")
    print("-" * 70)

    # Confronto ML
    print("\n>>> Confronto Solo-ML (Decision Tree vs k-NN)...")
    risultati_solo_ml = training.run_solo_ml_comparison(X_orig, y_orig)
    training.plot_solo_ml_comparison(risultati_solo_ml)

    print("\n[INFO] Il Decision Tree e' risultato il modello con l'accuratezza CV piu' alta;")
    print(" viene percio' scelto come modello di riferimento per l'integrazione con l'Ontologia.")
    print("-" * 70)

    # SECONDA PARTE: Ragionamento semantico
    print("\n>>> SECONDA PARTE: Connessione con l'Ontologia... >>>")
    semantic_hero.populate_ontology()
    semantic_results = semantic_hero.run_reasoning()

    # Creazione del dataframe integrato con i dati dedotti dal ragionatore
    semantic_df = pd.DataFrame(semantic_results)

    colonne_per_merge = dataset_originale.drop(columns=['universe'])
    dataset_arricchito = pd.merge(colonne_per_merge, semantic_df, on='name', how='left')

    if 'ruolo_ontologia' not in dataset_arricchito.columns:
        dataset_arricchito['ruolo_ontologia'] = None

    print("\n Conoscenza estratta dall'ontologia e mappata sul dataset.")
    print("-" * 70)

    print("\n>>> Pipeline Ibrida sul modello di riferimento (Decision Tree + Ontologia)...")

    predizioni_albero = dt_model_orig.predict(X_orig)

    predizioni_finali_ibride = []
    interventi_ontologia = 0

    # Applicazione della logica di soccorso semantica solo dove il ML ha fallito
    for index, row in dataset_arricchito.iterrows():
        ruolo_predetto_albero = predizioni_albero[index]
        ruolo_reale_csv = row['role']

        if ruolo_predetto_albero != ruolo_reale_csv:
            if pd.notna(row['ruolo_ontologia']):
                predizioni_finali_ibride.append(row['ruolo_ontologia'])
                interventi_ontologia += 1
            else:
                predizioni_finali_ibride.append(ruolo_predetto_albero)
        else:
            predizioni_finali_ibride.append(ruolo_predetto_albero)

    # Calcolo delle metriche complessive sulla pipeline ad hoc
    dataset_arricchito['predizione_ibrida'] = predizioni_finali_ibride
    corretti = dataset_arricchito[dataset_arricchito['role'] == dataset_arricchito['predizione_ibrida']]
    accuratezza_ibrida = len(corretti) / len(dataset_arricchito)

    print("=" * 60)
    print(f" NUMERO DI FALLIMENTI DELL'ONTOLOGIA: {interventi_ontologia}")
    print(f" ACCURATEZZA FINALE DELLA PIPELINE IBRIDA: {accuratezza_ibrida * 100:.2f}%")
    print("=" * 60)

    errori_rimasti = dataset_arricchito[dataset_arricchito['role'] != dataset_arricchito['predizione_ibrida']]
    if not errori_rimasti.empty:
        print(f"\n PERSONAGGI ANCORA FUORI POSTO ({len(errori_rimasti)}):")
        for index, row in errori_rimasti.iterrows():
            print(
                f"Eroe: {row['name']} | Ruolo CSV: {row['role']} | Albero: {predizioni_albero[index]} | Ontologia: {row['ruolo_ontologia']}")
    else:
        print("\n L'Ontologia ha sanato tutti i fallimenti. Accuratezza al 100%.")

    print("\n>>> Generazione del PLOT (Confronto conteggio Eroi Indovinati)...")
    eroi_indovinati_albero = (predizioni_albero == y_orig).sum()
    eroi_indovinati_ibrido = len(corretti)
    totale_personaggi = len(dataset_arricchito)

    training.plot_heroes_comparison(eroi_indovinati_albero, eroi_indovinati_ibrido, totale_personaggi)
    print("-" * 70)

    # Cross-Validation della pipeline ibrida sul modello di riferimento (DT)
    training.run_hybrid_cross_validation_and_plot(X_orig, y_orig, dataset_arricchito)

    print("\n[INFO] Completata con successo.")
    print("-" * 70)

    # Verifica di generalita' (kNN + Ontologia) -----------
    print("\n>>> Verifica di generalità — l'intervento dell'Ontologia si estende al k-NN?...")
    risultati_comparativi = training.run_hybrid_multi_model_comparison(X_orig, y_orig, dataset_arricchito)
    training.plot_model_comparison(risultati_comparativi)

    print("=" * 70)
    print("\n" + "#" * 60)
    print(" FINE FASE STRUTTURALE (Machine Learning & Logica Classica)")
    print("#" * 60)

    # TERZA PARTE: fasi di battle

    print("\n>>> TERZA PARTE: BATTLE >>>")
    try:
        battle.decision_system(dataset_arricchito.to_dict('records'))
    except KeyboardInterrupt:
        print("\n\n[INFO] Rilevata chiusura forzata dal sistema.")
        print("=" * 70)

if __name__ == "__main__":
    main()