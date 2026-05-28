import pandas as pd
from owlready2 import *

# Percorsi dinamici coerenti con la struttura del progetto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PATH_EMPTY = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_empty.owl')
PATH_POPULATED = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_populated.owl')
PATH_CSV = os.path.join(BASE_DIR, 'DataSet', 'super_heroes.csv')

# Caricamento dell'ontologia vuota, TBox, e creazione di un individuo per riga
# Assegnazione delle Data Properties qualitative grezze e
# salvataggio degli individui nell'ontologia.
def populate_ontology():
    print(f"[Ontologia] Caricamento file base da: {PATH_EMPTY}")
    onto = get_ontology(PATH_EMPTY).load()
    df = pd.read_csv(PATH_CSV)

    with onto:
        for _, row in df.iterrows():
            original_name = row['name'].replace(" ", "_")
            eroe = onto.Character(original_name)

            eroe.hasStrength = [int(row['strength'])]
            eroe.hasIntelligence = [int(row['intelligence'])]
            eroe.hasSpeed = [int(row['speed'])]
            eroe.hasPopularity = [int(row['popularity'])]

    onto.save(file=PATH_POPULATED, format="rdfxml")
    print(f" ---> Popolamento completato! Generati individui per {len(df)} eroi.")

# Esegue il ragionamento automatico tramite HermiT ed estrae i ruoli inferiti deduttivamente.
# Gestisce il fallback semantico (Specialist)
# Avvia il motore di inferenza HermiT
# Estrazione del ruolo basata sulla gerarchia inferita
def run_reasoning():
    onto = get_ontology(PATH_POPULATED).load()
    print("[Reasoner] Avvio del ragionatore HermiT sulle restrizioni sui tipi di dati...")

    with onto:
        sync_reasoner_hermit(infer_property_values=True)

    semantic_results = []
    for char in onto.Character.instances():
        original_name = char.name.replace("_", " ")
        classi_appartenenza = char.is_a

        if onto.Powerhouse in classi_appartenenza:
            inferred_role = 'Powerhouse'
        elif onto.Leader in classi_appartenenza:
            inferred_role = 'Leader'
        elif onto.Specialist in classi_appartenenza:
            inferred_role = 'Specialist'
        else:
            inferred_role = 'Specialist'

        semantic_results.append({
            'name': original_name,
            'ruolo_ontologia': inferred_role
        })

    print("---> Ragionamento completato. Nuova conoscenza semantica estratta con successo!")
    return semantic_results