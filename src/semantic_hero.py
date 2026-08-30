import os
import pandas as pd
from owlready2 import *

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))

PATH_EMPTY = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_empty.owl')
PATH_POPULATED = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_populated.owl')
PATH_CSV = os.path.join(BASE_DIR, 'DataSet', 'super_heroes.csv')


def populate_ontology():
    default_world.ontologies.clear()

    if not os.path.exists(PATH_EMPTY):
        raise FileNotFoundError(f"[ERRORE CRITICO] Ontologia non trovata in: {PATH_EMPTY}")

    print(f"[Ontologia] Caricamento file base da: {PATH_EMPTY}")
    onto = get_ontology(PATH_EMPTY).load()
    df = pd.read_csv(PATH_CSV)

    with onto:
        # Singole istanze condivise
        genio = onto.StrategicGenius("StrategicGenius_Single_Inst")
        forza = onto.DevastatingForce("DevastatingForce_Single_Inst")
        carisma = onto.Charismatic("Charismatic_Single_Inst")

        for _, row in df.iterrows():
            clean_name = row['name'].replace(" ", "_").replace("'", "").replace("-", "_")
            eroe = onto.Character(clean_name)

            str_val = int(row['strength'])
            intel_val = int(row['intelligence'])
            pop_val = int(row['popularity'])

            # Assegnazione Data Properties (Accettano liste di valori)
            eroe.hasStrength = [str_val]
            eroe.hasIntelligence = [intel_val]
            eroe.hasSpeed = [int(row['speed'])]
            eroe.hasPopularity = [pop_val]

            # GESTIONE CORRETTA OBJECT PROPERTIES (Senza liste vuote)
            if intel_val >= 8:
                eroe.hasTacticalProfile = genio
            elif str_val >= 7:
                eroe.hasTacticalProfile = forza
            else:
                eroe.hasTacticalProfile = None

            if pop_val >= 8:
                eroe.hasReputation = carisma
            else:
                eroe.hasReputation = None

    onto.save(file=PATH_POPULATED, format="rdfxml")
    print(f" ---> Popolamento completato! Generati individui per {len(df)} eroi.")


def run_reasoning():
    default_world.ontologies.clear()

    if not os.path.exists(PATH_POPULATED):
        raise FileNotFoundError(f"[ERRORE CRITICO] File popolato non trovato in: {PATH_POPULATED}")

    onto = get_ontology(PATH_POPULATED).load()
    print("[Reasoner] Avvio del ragionatore HermiT...")

    with onto:
        sync_reasoner_hermit(infer_property_values=True)

    semantic_results = []

    for char in onto.Character.instances():
        original_name = char.name.replace("_", " ")

        classi_inferite = char.INDIRECT_is_a

        if onto.Leader in classi_inferite:
            inferred_role = 'Leader'
        elif onto.Powerhouse in classi_inferite:
            inferred_role = 'Powerhouse'
        else:
            inferred_role = 'Specialist'

        semantic_results.append({
            'name': original_name,
            'ruolo_ontologia': inferred_role
        })

    print("---> Ragionamento completato! Nuova conoscenza semantica estratta con successo.")
    return semantic_results