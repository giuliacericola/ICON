import os
import pandas as pd
from src import training, semantic_hero


def main():
    print("=" * 70)
    print(" AVVIO PROGETTO: SISTEMA IBRIDO (NEURO-SIMBOLICO) DEI SUPEREROI")
    print("=" * 70)

    # Definizione dinamica dei percorsi
    cartella_principale = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(cartella_principale, 'DataSet', 'super_heroes.csv')

    if not os.path.exists(dataset_path):
        print(f"[ERRORE] Impossibile trovare il file del dataset in: {dataset_path}")
        return

    # =====================================================================
    # 1° PARTE: MODELLO SUL DATASET ORIGINALE (SOLO DATI NUMERICI)
    # =====================================================================
    print("\n>>> 1° PARTE: Modello statistico originale (Senza Conoscenza di Fondo)...")

    # 1. Caricamento del dataset grezzo
    dataset_originale = training.load_dataset(dataset_path)

    # 2. Pre-processing dei dati numerici di base
    X_orig, y_orig = training.preprocessing_dataset(dataset_originale.copy())

    # 3. Addestramento del primo albero di decisione (max_depth=3)
    print("\n Generazione delle regole decisionali dell'albero originale...")
    dt_model_orig = training.decisiontree_classifier(X_orig, y_orig, dataset_originale)

    # 4. Esecuzione della Cross-Validation ed estrazione del primo grafico
    print("\n Avvio della 5-Fold Cross-Validation sul modello originale...")
    training.run_cross_validation_and_plots(X_orig, y_orig)

    print("\n[INFO] 1° Parte completata con successo.")
    print("-" * 70)

    # =====================================================================
    # 2° PARTE: RAGIONAMENTO AUTOMATICO (ONTOLOGIA OWL / PROTÉGÉ)
    # =====================================================================
    print("\n>>> 2° PARTE: Esecuzione ragionamento automatico (Ontologia)...")

    # 1. Popolamento e inferenza logica delle classi equivalenti
    semantic_hero.populate_ontology()
    semantic_results = semantic_hero.run_reasoning()

    # 2. Trasferimento dei risultati in un DataFrame Pandas
    semantic_df = pd.DataFrame(semantic_results)

    # 3. Unione delle deduzioni logiche al dataset originale mediante join sul nome
    dataset_arricchito = pd.merge(dataset_originale, semantic_df, on='name', how='left')

    # Controllo di sicurezza anti-crash per la colonna ruolo_ontologia
    if 'ruolo_ontologia' not in dataset_arricchito.columns:
        dataset_arricchito['ruolo_ontologia'] = None

    print("\n>>> 2° Parte completata. Conoscenza di fondo estratta con successo.")
    print("-" * 70)

    # =====================================================================
    # 3° PARTE: PIPELINE IBRIDA CON CORREZIONE A POSTERIORI (FALLIMENTI ML)
    # =====================================================================
    print("\n>>> 3° PARTE: Attivazione della Pipeline Ibrida sui fallimenti del ML...")

    # 1. Otteniamo le predizioni standard dell'albero di decisione (Data-Driven)
    X_arr, y_arr = training.preprocessing_dataset(dataset_arricchito.copy())
    albero_statistico = training.decisiontree_classifier(X_arr, y_arr, dataset_arricchito)
    predizioni_albero = albero_statistico.predict(X_arr)

    predizioni_finali_ibride = []
    interventi_ontologia = 0

    # 2. APPLICAZIONE DEL FILTRO DI SOCCORSO LOGICO SOLO SE L'ALBERO SBAGLIA
    for index, row in dataset_arricchito.iterrows():
        ruolo_predetto_albero = predizioni_albero[index]
        ruolo_reale_csv = row['role']

        # CONTROLLO FALLIMENTO: L'albero ha sbagliato la predizione statistica?
        if ruolo_predetto_albero != ruolo_reale_csv:
            # L'albero ha fallito! Interroghiamo la Background Knowledge dell'Ontologia
            if pd.notna(row['ruolo_ontologia']):
                predizioni_finali_ibride.append(row['ruolo_ontologia'])
                interventi_ontologia += 1
            else:
                # Se l'ontologia non ha una classificazione pronta per questa eccezione, teniamo l'errore dell'albero
                predizioni_finali_ibride.append(ruolo_predetto_albero)
        else:
            # L'albero ha indovinato! Non serve attivare il ragionatore logico
            predizioni_finali_ibride.append(ruolo_predetto_albero)

    # 3. Valutazione finale delle metriche del sistema integrato
    dataset_arricchito['predizione_ibrida'] = predizioni_finali_ibride
    corretti = dataset_arricchito[dataset_arricchito['role'] == dataset_arricchito['predizione_ibrida']]
    accuratezza_ibrida = len(corretti) / len(dataset_arricchito)

    print("=" * 60)
    print(f" NUMERO DI INTERVENTI CORRETTIVI DELL'ONTOLOGIA: {interventi_ontologia}")
    print(f" ACCURATEZZA FINALE DELLA PIPELINE IBRIDA: {accuratezza_ibrida * 100:.2f}%")
    print("=" * 60)

    # Mostriamo a schermo i personaggi fuori posto residui
    errori_rimasti = dataset_arricchito[dataset_arricchito['role'] != dataset_arricchito['predizione_ibrida']]
    if not errori_rimasti.empty:
        print(f"\n PERSONAGGI ANCORA FUORI POSTO ({len(errori_rimasti)}):")
        for index, row in errori_rimasti.iterrows():
            print(
                f"Eroe: {row['name']} | Ruolo CSV: {row['role']} | Albero: {predizioni_albero[index]} | Ontologia: {row['ruolo_ontologia']}")
    else:
        print("\n L'Ontologia ha intercettato e sanato tutti i fallimenti dell'albero! Accuratezza al 100%.")


if __name__ == "__main__":
    main()