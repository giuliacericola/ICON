import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Funzione di supporto per evitare crash in caso di valori NaN o nulli nel dataset
def support(value, default=5):
    try:
        if value is None or str(value).lower() == 'nan' or str(value).strip() == '':
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

# Modulo avanzato di Question Answering e Simulazione di Battaglia Strategica.
# Riceve il dataset arricchito dal main, proietta la conoscenza nello Spazio
# delle Feature e calcola le affinità semantiche tramite Coseno.
# Recuperato il ruolo stabilito dall'ontologia (estratto nella parte precedente del progetto)
# Traduzione controllata delle statistiche dei supereroi in vincoli logici, al riparo da errori
# di formattazione del dataset.
# Vettorizzazione ed embedding, definiti gli scenari di query
# Calcolo del coseno come euristica di matching
# Stampa i risultati
def query_responding(dataset_completo):

    semantic_body = {}
    heroes_map = {}

    for row in dataset_completo:
        hero_name = row['name']
        universe_raw = 'Unknown'
        for key_universe in ['universe', 'universe_x', 'universe_y', 'faction', 'Faction', 'universo']:
            if key_universe in row and row[key_universe] is not None and str(row[key_universe]) != 'nan':
                universe_raw = str(row[key_universe]).strip()
                break

        if universe_raw.upper() in ['MC', 'MARVEL', 'MARVEL COMICS']:
            universe = "Marvel"
        elif universe_raw.upper() in ['DC', 'DC COMICS']:
            universe = "DC"
        else:
            universe = universe_raw

        predict_class = row.get('ruolo_ontologia', 'Character')
        if predict_class is None or str(predict_class) == 'nan' or str(predict_class).strip() == '':
            predict_class = 'Character'

        tracts = []
        pop = support(row.get('popularity'))
        speed = support(row.get('speed'))
        strg = support(row.get('strength'))
        intel = support(row.get('intelligence'))
        powers = support(row.get('power_source'))

        if pop <= 4:
            tracts.append("has_trait_low_profile")
        elif pop >= 8:
            tracts.append("has_trait_influencer")

        if speed >= 7:
            tracts.append("has_trait_high_mobility")

        if strg >= 8:
            tracts.append("has_trait_heavy_hitter")

        if intel >= 8:
            tracts.append("has_trait_tactician")

        if powers == 'Technological_Weapon':
            tracts.append("has_trait_TechonologicalWeapon")


        string_trats = " ".join(tracts)

        string = f"Character_{hero_name} is_classified_as_{predict_class} belongs_to_{universe} {string_trats}"

        semantic_body[hero_name] = string
        heroes_map[hero_name] = {
            "nome": hero_name.replace("_", " "),
            "ruolo": predict_class,
            "universo": universe
        }

    # Proiezione nello Spazio delle Feature
    heroes_names = list(semantic_body.keys())
    semantic_text = list(semantic_body.values())

    vectorizer = TfidfVectorizer()
    mat_embedding = vectorizer.fit_transform(semantic_text)

    scenary_query = {
        "1": {
            "titolo": "MISSIONE SEGRETA (Infiltrazione spionistica)",
            "descrizione": "Seleziona eroi Specialist a basso profilo e alta mobilità.",
            "stringa_target": "is_classified_as_Specialist has_trait_low_profile has_trait_high_mobility"
        },
        "2": {
            "titolo": "SIMULATORE DI BATTAGLIA: SQUADRA MARVEL vs MINACCIA DC",
            "descrizione": "Genera la coalizione ottimale Marvel (Powerhouse e Leader tattici) per contrastare la DC.",
            "stringa_target": "belongs_to_Marvel is_classified_as_Powerhouse has_trait_heavy_hitter is_classified_as_Leader has_trait_tactician"
        },
        "3": {
            "titolo": "LEADER IDEALE (Attacco Cyber)",
            "descrizione": "Chi rispecchia maggiormente il ruolo di leader per fronteggiare un attacco cyber",
            "stringa_target": "is_classified_as_Leader has_trait_tactician has_trait_TechonologicalWeapon"
        }

    }


    while True:
        print("\n" + "=" * 75)
        print("QUESTION ANSWERING ")
        print("=" * 75)
        print("Scegli lo scenario!")
        for key, item in scenary_query.items():
            print(f" [{key}] {item['titolo']}\n     -> {item['descrizione']}")
        print(" [0] Esci dal programma")

        choose = input("\nInserisci il numero dello scenario desiderato: ").strip()

        if choose == "0":
            print("\nChiusura del modulo di Question Answering")
            print("=" * 75)
            break

        if choose not in scenary_query:
            print("[INFO] Scelta non valida. Riprova.")
            continue

        chosen_scenary = scenary_query[choose]
        right_query = chosen_scenary["stringa_target"]

        # Calcolo coseno
        vect = vectorizer.transform([right_query])
        sim = cosine_similarity(vect, mat_embedding).flatten()
        index = np.argsort(sim)[::-1]

        # stampa ris
        print("\n" + "-" * 75)
        print(f" RISPOSTA ALLA QUERY SEMANTICA: {chosen_scenary['titolo']}")
        print(f" Target: '{right_query}'")
        print("-" * 75)

        if choose == "1":
            print(" >> INFILTRAZIONE: Rilevato scenario ostile ad alto rischio visibilità.")
            print(" >> ALGORITMO: Estrazione agenti furtivi...\n")
            print(f" {'Membro Suggerito':<25} | {'Classe OWL':<15} | {'Universo':<10} | {'Affinità'}")
        elif choose == "2":
            print(" >> ALLERTA: Squadra nemica DC Comics in avvicinamento.")
            print(" >> ALGORITMO: Ottimizzazione della coalizione difensiva Marvel...\n")
            print(f" {'Membro Reclutato':<25} | {'Classe OWL':<15} | {'Universo':<10} | {'Affinità'}")
        elif choose == "3":
            print(" >> CYBER ATTACK: Violazione dell'infrastruttura di rete rilevata.")
            print(" >> ALGORITMO: Selezione di vertici di comando con feature tecnologiche...\n")
            print(f" {'Leader Consigliato':<25} | {'Classe OWL':<15} | {'Universo':<10} | {'Affinità'}")
        print("-" * 75)

        c = 0
        for idx in index:
            score = sim[idx]
            id_hero = heroes_names[idx]
            info = heroes_map[id_hero]

            # Filtro asimmetrico per la battaglia: escludiamo i difensori che non appartengono alla Marvel
            if choose == "2" and info['universo'] != "Marvel":
                continue

            if score > 0 and c < 4:
                print(f" -> {info['nome']:<22} | {info['ruolo']:<15} | {info['universo']:<10} | {score * 100:.2f}%")
                c += 1

        if choose == "1":
            print("-" * 75)
            print(" ESITO SIMULAZIONE: Task-force a basso profilo identificata tramite minimizzazione d'angolo.")
            print(" Parametri di mobilità e anonimato geometricamente soddisfatti.")
        elif choose == "2":
            print("-" * 75)
            print(" ESITO SIMULAZIONE: Coerenza tattica della coalizione Marvel ottimizzata.")
            print(" Spazio degli Stati potato con successo tramite euristica geometrica.")
        elif choose == "3":
            print("-" * 75)
            print(" ESITO SIMULAZIONE: Comando di difesa cyber strutturato con successo.")
            print(" Intersezione di Feature logiche e armamenti tecnologici massimizzata.")

        print("=" * 75)

        input("\nPremere [INVIO] per tornare al menu delle query...")