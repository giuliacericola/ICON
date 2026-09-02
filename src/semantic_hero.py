import pandas as pd
from owlready2 import *

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))

PATH_EMPTY = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_empty.owl')
PATH_POPULATED = os.path.join(BASE_DIR, 'Ontologia', 'super_heroes_populated.owl')
PATH_CSV = os.path.join(BASE_DIR, 'DataSet', 'super_heroes.csv')

# Metodo che popola l'ontologia.
# Carica l'ontologia base vuota (TBox) e il dataset CSV, creando le istanze OWL
# per ciascun eroe e per le background-knowledge.
# Inserisce unicamente i dati grezzi senza prendere decisioni logiche,
# delegando ogni classificazione al reasoner HermiT.
def populate_ontology():
    default_world.ontologies.clear()

    if not os.path.exists(PATH_EMPTY):
        raise FileNotFoundError(f"[ERRORE CRITICO] Ontologia non trovata in: {PATH_EMPTY}")

    print(f"[Ontologia] Caricamento file base da: {PATH_EMPTY}")
    onto = get_ontology(PATH_EMPTY).load()
    df = pd.read_csv(PATH_CSV)

    with onto:
        supernatural = onto.Supernatural("Supernatural_Single_Inst")
        tech_weapon = onto.TechnologicalWeapon("TechnologicalWeapon_Single_Inst")
        emp = onto.EMP("EMP_Single_Inst")
        sigillo = onto.AntiMagicSeal("AntiMagicSeal_Single_Inst")

        # Background knowledge asserita una sola volta: il reasoner deduce
        # hasVulnerability di ogni eroe componendo hasPowerSource + hasWeakness.
        tech_weapon.hasWeakness = [emp]
        supernatural.hasWeakness = [sigillo]

        for _, row in df.iterrows():
            raw_name = str(row['name'])
            clean_name = raw_name.replace(" ", "_").replace("'", "").replace("-", "_")
            eroe = onto.Character(clean_name)

            # Assegnazione rdfs:label per preservare esattamente il nome del CSV
            eroe.label = [raw_name]

            # Caricamento dati grezzi
            eroe.hasStrength = [int(row['strength'])]
            eroe.hasIntelligence = [int(row['intelligence'])]
            eroe.hasSpeed = [int(row['speed'])]
            eroe.hasPopularity = [int(row['popularity'])]

            power_raw = str(row.get('power_source', '')).strip()
            if power_raw == 'Technological_Weapon':
                eroe.hasPowerSource = [tech_weapon]
            else:
                eroe.hasPowerSource = [supernatural]

    onto.save(file=PATH_POPULATED, format="rdfxml")
    print(f" ---> Popolamento completato! Generati individui per {len(df)} eroi.")

# INFERENZA LOGICA E RAGIONAMENTO SEMANTICO
# Carica l'ontologia popolata, esegue il ragionatore HermiT per classificare
# automaticamente gli eroi e risolvere le property chain (es. vulnerabilità EMP).
# Estrae infine la conoscenza dedotta sotto forma di
# dizionari pronti per la fase successiva.
def run_reasoning():
    default_world.ontologies.clear()

    if not os.path.exists(PATH_POPULATED):
        raise FileNotFoundError(f"[ERRORE CRITICO] File popolato non trovato in: {PATH_POPULATED}")

    onto = get_ontology(PATH_POPULATED).load()
    print("[Reasoner] Avvio del ragionatore HermiT...")

    with onto:
        with onto:
            sync_reasoner_hermit(debug=0)

    semantic_results = []

    for char in onto.Character.instances():
        original_name = char.label[0] if char.label else char.name.replace("_", " ")

        classi_inferite = char.INDIRECT_is_a

        if onto.Leader in classi_inferite:
            inferred_role = 'Leader'
        elif onto.Powerhouse in classi_inferite:
            inferred_role = 'Powerhouse'
        else:
            inferred_role = 'Specialist'

        # Tratti estratti dalle classi ontologiche inferite da HermiT
        is_high_mobility = onto.HighMobility in classi_inferite
        is_heavy_hitter = onto.HeavyHitter in classi_inferite
        is_tactician = onto.StrategicGenius in classi_inferite
        is_low_profile = onto.LowProfile in classi_inferite
        is_influencer = onto.Charismatic in classi_inferite
        is_tech = onto.TechnologicalWeapon_Single_Inst in char.hasPowerSource

        semantic_results.append({
            'name': original_name,
            'ruolo_ontologia': inferred_role,
            'is_high_mobility': is_high_mobility,
            'is_heavy_hitter': is_heavy_hitter,
            'is_tactician': is_tactician,
            'is_low_profile': is_low_profile,
            'is_influencer': is_influencer,
            'is_tech': is_tech
        })

    print("---> Ragionamento completato! Nuova conoscenza semantica estratta con successo.")

    # Deduce EMPTarget componendo
    # la property chain (hasPowerSource + hasWeakness) senza assegnazioni dirette.
    bersagli_emp = [c.label[0] if c.label else c.name.replace("_", " ") for c in onto.EMPTarget.instances()]
    print(f"[Deduzione] Eroi dedotti vulnerabili a EMP (mai assegnati esplicitamente): {len(bersagli_emp)}")

    return semantic_results