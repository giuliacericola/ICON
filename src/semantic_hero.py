import os
import pandas as pd
from owlready2 import *

# Percorsi dinamici coerenti con la struttura del progetto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PATH_EMPTY = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_empty.owl')
PATH_POPULATED = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_populated.owl')
PATH_CSV = os.path.join(BASE_DIR, 'DataSet', 'super_heroes.csv')


def populate_ontology():
    """
    Carica l'ontologia vuota (TBox) e crea un individuo per ogni riga del dataset,
    assegnando le data properties numeriche grezze necessarie all'inferenza.
    """
    print(f"[Ontologia] Caricamento file base da: {PATH_EMPTY}")
    onto = get_ontology(PATH_EMPTY).load()
    df = pd.read_csv(PATH_CSV)

    with onto:
        for _, row in df.iterrows():
            # Pulizia del nome per renderlo un URI valido nell'ABox
            nome_pulito = row['name'].replace(" ", "_")
            eroe = onto.Character(nome_pulito)

            # Assegnazione delle Data Properties qualitative grezze
            eroe.hasStrength = [int(row['strength'])]
            eroe.hasIntelligence = [int(row['intelligence'])]
            eroe.hasSpeed = [int(row['speed'])]
            eroe.hasPopularity = [int(row['popularity'])]

    # Salvataggio dell'ABox popolata
    onto.save(file=PATH_POPULATED, format="rdfxml")
    print(f" -> Popolamento completato! Generati individui per {len(df)} eroi.")


def run_reasoning():
    """
    Esegue il ragionamento automatico tramite HermiT ed estrae i ruoli inferiti deduttivamente.
    Gestisce il fallback semantico (Specialist) a livello applicativo per preservare l'OWA.
    """
    onto = get_ontology(PATH_POPULATED).load()
    print("[Reasoner] Avvio del ragionatore HermiT sulle restrizioni sui tipi di dati...")

    with onto:
        # Avvio del motore di inferenza HermiT
        sync_reasoner_hermit(infer_property_values=True)

    semantic_results = []
    for char in onto.Character.instances():
        name_originale = char.name.replace("_", " ")
        classi_appartenenza = char.is_a

        # Estrazione del ruolo basata sulla gerarchia inferita
        ruolo_inferito = None
        if onto.Powerhouse in classi_appartenenza:
            ruolo_inferito = 'Powerhouse'
        elif onto.Leader in classi_appartenenza:
            ruolo_inferito = 'Leader'
        elif onto.Specialist in classi_appartenenza:
            ruolo_inferito = 'Specialist'
        else:
            # Gestione del fallback per esclusione a livello applicativo (Modello "Episodi Buoni" della relazione)
            ruolo_inferito = 'Specialist'

        semantic_results.append({
            'name': name_originale,
            'ruolo_ontologia': ruolo_inferito
        })

    print(" -> Ragionamento completato. Nuova conoscenza semantica estratta con successo.")
    return semantic_results