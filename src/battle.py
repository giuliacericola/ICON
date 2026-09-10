# Modulo di Simulazione di Battaglia Strategica con Reasoning a Due Livelli.
# Primo livello: Deduzione Logica Esatta (OWL DL EquivalentClasses / Property Chains).
# Secondo livello: Ranking Euristico di Prossimità (Fallback per copertura parziale dei criteri).
def decision_system(dataset_completo):
    heroes_map = {}
    for row in dataset_completo:
        heroes_map[row['name']] = {
            "nome": str(row['name']),
            "ruolo": row.get('ruolo_ontologia', 'Character'),
            "universo": row.get('universe', 'Unknown'),

            # Classi composte dedotte direttamente dal Reasoner (TBox)
            "is_ideal_infiltrator": bool(row.get('is_ideal_infiltrator', False)),
            "is_ideal_cyber_commander": bool(row.get('is_ideal_cyber_commander', False)),
            "is_emp_target": bool(row.get('is_emp_target', False)),
            "is_antimagic_target": bool(row.get('is_antimagic_target', False)),

            # Criteri booleani elementari (usati per il fallback euristico)
            "is_specialist": row.get('ruolo_ontologia') == 'Specialist',
            "is_powerhouse": row.get('ruolo_ontologia') == 'Powerhouse',
            "is_leader": row.get('ruolo_ontologia') == 'Leader',
            "is_low_profile": bool(row.get('is_low_profile', False)),
            "is_high_mobility": bool(row.get('is_high_mobility', False)),
            "is_heavy_hitter": bool(row.get('is_heavy_hitter', False)),
            "is_tactician": bool(row.get('is_tactician', False)),
            "is_tech": bool(row.get('is_tech', False)),
        }

    scenary_query = {
        "1": {
            "titolo": "MISSIONE SEGRETA (Infiltrazione spionistica)",
            "descrizione": "Estrazione diretta della classe composta OWL 'IdealInfiltrator' (Specialist ⊓ LowProfile ⊓ HighMobility).",
            "classe_dedotta": "is_ideal_infiltrator",
            "criteri_fallback": ["is_specialist", "is_low_profile", "is_high_mobility"]
        },
        "2": {
            "titolo": "SIMULATORE DI BATTAGLIA: SQUADRA MARVEL vs MINACCIA DC",
            "descrizione": "Genera la coalizione ottimale Marvel valutando sia profilazione esatta sia copertura parziale dei ruoli.",
            "classe_dedotta": None,  # Mantiene selezione di squadra per criteri su vincolo di universo
            "criteri_fallback": ["is_powerhouse", "is_heavy_hitter", "is_leader", "is_tactician"],
            "vincolo_universo": "Marvel"
        },
        "3": {
            "titolo": "LEADER IDEALE (Attacco Cyber)",
            "descrizione": "Estrazione diretta della classe composta OWL 'IdealCyberCommander' (Leader ⊓ ∃hasPowerSource.TechnologicalWeapon).",
            "classe_dedotta": "is_ideal_cyber_commander",
            "criteri_fallback": ["is_leader", "is_tactician", "is_tech"]
        },
        "4": {
            "titolo": "CONTROMISURA EMP (Bersagli vulnerabili)",
            "descrizione": "Identifica le istanze della classe dedotta EMPTarget via Property Chain (hasPowerSource ∘ hasWeakness ⊑ hasVulnerability).",
            "classe_dedotta": "is_emp_target",
            "criteri_fallback": ["is_emp_target"]
        },
        "5": {
            "titolo": "CONTROMISURA ANTI-MAGICA (Bersagli vulnerabili)",
            "descrizione": "Identifica le istanze della classe dedotta AntiMagicSealTarget via Property Chain.",
            "classe_dedotta": "is_antimagic_target",
            "criteri_fallback": ["is_antimagic_target"]
        }
    }

    while True:
        print("\n" + "=" * 75)
        print("QUESTION ANSWERING SEMANTICO (Reasoning DL Esatto + Fallback Euristico)")
        print("=" * 75)
        print("Scegli lo scenario!")
        for key, item in scenary_query.items():
            print(f" [{key}] {item['titolo']}\n     -> {item['descrizione']}")
        print(" [0] Esci dal programma")

        choose = input("\nInserisci il numero dello scenario desiderato: ").strip()

        if choose == "0":
            print("\nChiusura del modulo Battle")
            print("=" * 75)
            break

        if choose not in scenary_query:
            print("[INFO] Scelta non valida. Riprova.")
            continue

        chosen = scenary_query[choose]
        classe_dedotta = chosen.get("classe_dedotta")
        criteri_fallback = chosen["criteri_fallback"]
        vincolo_universo = chosen.get("vincolo_universo")

        candidati_esatti = []
        candidati_parziali = []

        # --- FASE 1: RAGIONAMENTO LOGICO ESATTO (TBox EquivalentClasses / Reasoning) ---
        for nome, info in heroes_map.items():
            if vincolo_universo and info["universo"] != vincolo_universo:
                continue

            # Se lo scenario prevede una classe composta equivalente, chiediamo la membership al reasoner
            if classe_dedotta and info.get(classe_dedotta, False):
                candidati_esatti.append((1.0, len(criteri_fallback), nome, info, "DEDUZIONE LOGICA ESATTA (100%)"))
            else:
                # --- FASE 2: COPERTURA EURISTICA PARZIALE (Fallback) ---
                soddisfatti = sum(1 for c in criteri_fallback if info.get(c, False))
                copertura = soddisfatti / len(criteri_fallback)
                if copertura > 0:
                    candidati_parziali.append(
                        (copertura, soddisfatti, nome, info, f"PROSSIMITÀ EURISTICA ({copertura * 100:.0f}%)"))

        # Selezione della strategia di output
        if candidati_esatti:
            # Ordinamento alfabetico sui candidati esatti per riproducibilità deterministica
            candidati_esatti.sort(key=lambda t: t[2])
            risultati = candidati_esatti
            modalita_risposta = "REASONER DL (Corrispondenza formale perfetta trovata in TBox)"
        else:
            candidati_parziali.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            risultati = candidati_parziali
            modalita_risposta = "FALLBACK EURISTICO (Nessun match esatto in TBox, ranking per prossimità)"

        print("\n" + "-" * 75)
        print(f" RISPOSTA ALLA QUERY SEMANTICA: {chosen['titolo']}")
        print(f" MODALITÀ ELABORAZIONE: {modalita_risposta}")
        print("-" * 75)

        intestazioni = {
            "1": "Membro Suggerito",
            "2": "Membro Reclutato",
            "3": "Leader Consigliato",
            "4": "Bersaglio EMP",
            "5": "Bersaglio Anti-Magia"
        }
        etichetta = intestazioni[choose]

        print(f" {etichetta:<22} | {'Classe OWL':<15} | {'Universo':<10} | {'Esito Reasoning / Match'}")
        print("-" * 75)

        for copertura, soddisfatti, nome, info, tipo_match in risultati[:5]:
            print(f" -> {info['nome']:<19} | {info['ruolo']:<15} | {info['universo']:<10} | {tipo_match}")

        print("-" * 75)
        esiti = {
            "1": " ESITO: Infiltrati identificati direttamente tramite la classe composta IdealInfiltrator.",
            "2": " ESITO: Coalizione Marvel ottimizzata tramite intersezione dei profili tattici.",
            "3": " ESITO: Cyber-comandanti identificati tramite la classe composta IdealCyberCommander.",
            "4": " ESITO: Bersagli EMP dedotti tramite la property chain hasPowerSource ∘ hasWeakness.",
            "5": " ESITO: Bersagli Anti-Magia dedotti tramite la property chain hasPowerSource ∘ hasWeakness."
        }
        print(esiti[choose])
        print("=" * 75)

        input("\nPremere [INVIO] per tornare al menu delle query...")