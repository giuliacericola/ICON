import os
from src import training


def main():
    print("=" * 65)
    print(" AVVIO PROGETTO: ESTRATTORE SEMANTICO E BATTLE ENGINE DEI SUPEREROI")
    print("=" * 65)

    # PERCORSO CORRETTO: calcolato partendo dalla posizione di main.py
    cartella_principale = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(cartella_principale, 'DataSet', 'super_heroes.csv')

    # Controllo visivo per sicurezza
    if not os.path.exists(dataset_path):
        print(f"[ERRORE] Impossibile trovare il file in: {dataset_path}")
        return

    # =====================================================================
    # 1° PARTE: ADDESTRAMENTO E VALIDAZIONE (Machine Learning)
    # =====================================================================
    print("\n>>> 1° PARTE: Estrazione delle regole logiche dal dataset...")

    # 1. Caricamento del dataset
    dataset_originale = training.load_dataset(dataset_path)

    # 2. Pre-processing
    X, y = training.preprocessing_dataset(dataset_originale.copy())

    # 3. Addestramento dell'albero (max_depth=3)
    print("\n Generazione delle regole decisionali dell'albero (Profondità 3)...")
    dt_model = training.decisiontree_classifier(X, y, dataset_originale)

    # 4. Validazione incrociata e generazione del grafico visivo
    print("\n Avvio della 5-Fold Cross-Validation per testare la tenuta del modello...")
    training.run_cross_validation_and_plots(X, y)

    print("\n[INFO] 1° Parte completata con successo. I ruoli sono stati estratti.")
    print("-" * 65)


if __name__ == "__main__":
    main()