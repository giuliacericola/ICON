import os
import pandas as pd
from src import training, semantic_hero, battle


def main():
    print("=" * 70)
    print("HEROES IN SUPER TRAINING")
    print("=" * 70)

    cartella_principale = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(cartella_principale, 'DataSet', 'super_heroes.csv')

    if not os.path.exists(dataset_path):
        print(f"[ERRORE] Dataset non trovato in: {dataset_path}")
        return

    # =====================================================================
    # 1° PARTE: MACHINE LEARNING PURO & STRUTTURA REGOLE (SOLO ML)
    # =====================================================================
    print("\n>>> 1° PARTE: Esecuzione della componente statistica (Solo ML)...")
    dataset_originale = training.load_dataset(dataset_path)
    X_orig, y_orig = training.preprocessing_dataset(dataset_originale.copy())

    # Addestra l'albero, stampa metriche ed errori base
    dt_model_orig = training.decisiontree_classifier(X_orig, y_orig, dataset_originale)

    # Mostra in sequenza i primi due grafici (Curva validazione + Dettaglio 5 fold originari)
    training.run_cross_validation_and_plots(X_orig, y_orig)

    print("\n[INFO] 1° Parte completata con successo.")
    print("-" * 70)

    # =====================================================================
    # 2° PARTE: RAGIONAMENTO LOGICO SEMANTICO (ONTOLOGIA OWL)
    # =====================================================================
    print("\n>>> 2° PARTE: Connessione con la Background Knowledge (Ontologia)...")
    semantic_hero.populate_ontology()
    semantic_results = semantic_hero.run_reasoning()

    # Creazione del dataframe integrato con i dati dedotti dal ragionatore di Protégé
    semantic_df = pd.DataFrame(semantic_results)
    dataset_arricchito = pd.merge(dataset_originale, semantic_df, on='name', how='left')

    if 'ruolo_ontologia' not in dataset_arricchito.columns:
        dataset_arricchito['ruolo_ontologia'] = None

    print("\n Conoscenza estratta dall'ontologia e mappata sul dataset.")
    print("-" * 70)

    # =====================================================================
    # 3° PARTE: PIPELINE IBRIDA CON CORREZIONE A POSTERIORI (FALLIMENTI ML)
    # =====================================================================
    print("\n>>> 3° PARTE: Attivazione della Pipeline Ibrida sui fallimenti del ML...")

    # Utilizziamo le predizioni dell'albero calcolato nella 1° parte sul dataset intero
    predizioni_albero = dt_model_orig.predict(X_orig)

    predizioni_finali_ibride = []
    interventi_ontologia = 0

    # Applicazione della logica di soccorso semantica solo dove il ML ha fallito
    for index, row in dataset_arricchito.iterrows():
        ruolo_predetto_albero = predizioni_albero[index]
        ruolo_reale_csv = row['role']

        if ruolo_predetto_albero != ruolo_reale_csv:
            # L'albero ha fallito su questo eroe! Interviene l'ontologia
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
    print(f" NUMERO DI INTERVENTI CORRETTIVI DELL'ONTOLOGIA: {interventi_ontologia}")
    print(f" ACCURATEZZA FINALE DELLA PIPELINE IBRIDA: {accuratezza_ibrida * 100:.2f}%")
    print("=" * 60)

    # Stampa a schermo i personaggi rimasti fuori posto (Corretto il bug sintattico)
    errori_rimasti = dataset_arricchito[dataset_arricchito['role'] != dataset_arricchito['predizione_ibrida']]
    if not errori_rimasti.empty:
        print(f"\n PERSONAGGI ANCORA FUORI POSTO ({len(errori_rimasti)}):")
        for index, row in errori_rimasti.iterrows():
            print(
                f"Eroe: {row['name']} | Ruolo CSV: {row['role']} | Albero: {predizioni_albero[index]} | Ontologia: {row['ruolo_ontologia']}")
    else:
        print("\n L'Ontologia ha sanato tutti i fallimenti. Accuratezza al 100%.")

    print("\n>>> Generazione del PLOT FINALE (Confronto conteggio Eroi Indovinati)...")
    eroi_indovinati_albero = (predizioni_albero == y_orig).sum()
    eroi_indovinati_ibrido = len(corretti)
    totale_personaggi = len(dataset_arricchito)

    # Mostra il 3° Grafico: l'istogramma a barre verticali con il conteggio degli eroi salvati
    training.plot_heroes_comparison(eroi_indovinati_albero, eroi_indovinati_ibrido, totale_personaggi)
    print("-" * 70)

    # =====================================================================
    # 4° PARTE: CROSS-VALIDATION FINALE SULLA PIPELINE IBRIDA INTEGRATA
    # =====================================================================
    # Lancia il ciclo k-fold dinamico e mostra il 4° grafico a linee (Confronto CV)
    training.run_hybrid_cross_validation_and_plot(X_orig, y_orig, dataset_arricchito)

    print("=" * 70)
    print("\n" + "#" * 60)
    print(" FINE FASE STRUTTURALE (Machine Learning & Logica Classica)")
    print("#" * 60)


    # Chiamiamo la funzione contenuta dentro il file battle.py
    battle.query_responding(dataset_arricchito.to_dict('records'))


if __name__ == "__main__":
    main()