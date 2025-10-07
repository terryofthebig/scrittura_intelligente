import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter
import os
import ssl

# Configurazione iniziale
st.set_page_config(
    page_title="Scrittura Intelligente",
    page_icon="✍️",
    layout="wide"
)

# Titolo e introduzione
st.title("✍️ App di Scrittura Intelligente")
st.markdown("""
Analizza il tuo stile di scrittura personale e ricevi strategie di miglioramento 
personalizzate basate sulle **intelligenze multiple** di Howard Gardner.
""")

# Tentativo di import NLTK con gestione errori
try:
    import nltk
    
    # Disabilita verifica SSL per evitare problemi di download
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Download delle risorse NLTK necessarie
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.tag import pos_tag
    
    nltk_available = True
    
except ImportError:
    st.error("⚠️ NLTK non è disponibile. L'analisi sarà limitata.")
    nltk_available = False
except Exception as e:
    st.warning(f"⚠️ Problema con NLTK: {str(e)}. Usando metodi alternativi.")
    nltk_available = False

# Sidebar per informazioni
with st.sidebar:
    st.header("Informazioni")
    st.markdown("""
    Questa app analizza:
    - **Vocabolario e complessità linguistica**
    - **Struttura delle frasi e paragrafi**
    - **Tono e stile narrativo**
    - **Coerenza e coesione testuale**
    """)
    
    if not nltk_available:
        st.error("Modalità limitata: NLTK non disponibile")

# Funzioni di fallback se NLTK non è disponibile
def tokenizza_testo_semplice(testo):
    """Tokenizzazione semplice senza NLTK"""
    return re.findall(r'\b\w+\b', testo.lower())

def conta_frasi_semplice(testo):
    """Conta frasi senza NLTK"""
    return len([f for f in re.split(r'[.!?]+', testo) if f.strip()])

def calcola_leggibilita_semplice(testo):
    """Calcola leggibilità senza NLTK"""
    parole = tokenizza_testo_semplice(testo)
    frasi = conta_frasi_semplice(testo)
    
    if len(parole) == 0 or frasi == 0:
        return 50
    
    lunghezza_media_frasi = len(parole) / frasi
    lunghezza_media_parole = sum(len(p) for p in parole) / len(parole)
    
    leggibilita = 100 - (lunghezza_media_frasi * 1.5) - (lunghezza_media_parole * 8)
    return max(0, min(100, leggibilita))

# Funzioni principali
def analizza_stile(testo):
    """Analizza vari aspetti dello stile di scrittura"""
    
    if nltk_available:
        try:
            # Usa NLTK se disponibile
            parole = word_tokenize(re.sub(r'[^\w\s]', '', testo))
            frasi_totali = len(sent_tokenize(testo))
            parole_totali = len(parole)
            
            # Analisi POS con NLTK
            tagged = pos_tag(parole)
            aggettivi = len([word for word, pos in tagged if pos in ['JJ', 'JJR', 'JJS']])
            avverbi = len([word for word, pos in tagged if pos in ['RB', 'RBR', 'RBS']])
            verbi = len([word for word, pos in tagged if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])
            sostantivi = len([word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']])
            
        except Exception:
            # Fallback a metodi semplici
            return analizza_stile_semplice(testo)
    else:
        # Usa metodi semplici
        return analizza_stile_semplice(testo)
    
    # Calcoli comuni
    lunghezza_media_frasi = parole_totali / frasi_totali if frasi_totali > 0 else 0
    vocab_unico = len(set(parole))
    ricchezza_lessicale = vocab_unico / parole_totali if parole_totali > 0 else 0
    leggibilita = calcola_leggibilita_semplice(testo)
    
    # Analisi del tono
    parole_pos_lista = ['buono', 'bello', 'fantastico', 'eccellente', 'meraviglioso', 'positivo', 'felice', 'gioia']
    parole_neg_lista = ['cattivo', 'brutto', 'terribile', 'orribile', 'pessimo', 'negativo', 'triste', 'dolore']
    
    parole_positive = sum(1 for word in parole if word.lower() in parole_pos_lista)
    parole_negative = sum(1 for word in parole if word.lower() in parole_neg_lista)
    tono_positivo = parole_positive / parole_totali if parole_totali > 0 else 0
    tono_negativo = parole_negative / parole_totali if parole_totali > 0 else 0
    
    return {
        'parole_totali': parole_totali,
        'frasi_totali': frasi_totali,
        'lunghezza_media_frasi': lunghezza_media_frasi,
        'vocab_unico': vocab_unico,
        'ricchezza_lessicale': ricchezza_lessicale,
        'aggettivi': aggettivi,
        'avverbi': avverbi,
        'verbi': verbi,
        'sostantivi': sostantivi,
        'leggibilita': leggibilita,
        'tono_positivo': tono_positivo,
        'tono_negativo': tono_negativo
    }

def analizza_stile_semplice(testo):
    """Analisi semplificata senza NLTK"""
    parole = tokenizza_testo_semplice(testo)
    frasi_totali = conta_frasi_semplice(testo)
    parole_totali = len(parole)
    
    lunghezza_media_frasi = parole_totali / frasi_totali if frasi_totali > 0 else 0
    vocab_unico = len(set(parole))
    ricchezza_lessicale = vocab_unico / parole_totali if parole_totali > 0 else 0
    leggibilita = calcola_leggibilita_semplice(testo)
    
    # Stime approssimative per parti del discorso
    aggettivi = len(parole) // 10
    avverbi = len(parole) // 20
    verbi = len(parole) // 5
    sostantivi = len(parole) // 3
    
    # Analisi del tono
    parole_pos_lista = ['buono', 'bello', 'fantastico', 'eccellente', 'meraviglioso', 'positivo', 'felice', 'gioia']
    parole_neg_lista = ['cattivo', 'brutto', 'terribile', 'orribile', 'pessimo', 'negativo', 'triste', 'dolore']
    
    parole_positive = sum(1 for word in parole if word in parole_pos_lista)
    parole_negative = sum(1 for word in parole if word in parole_neg_lista)
    tono_positivo = parole_positive / parole_totali if parole_totali > 0 else 0
    tono_negativo = parole_negative / parole_totali if parole_totali > 0 else 0
    
    return {
        'parole_totali': parole_totali,
        'frasi_totali': frasi_totali,
        'lunghezza_media_frasi': lunghezza_media_frasi,
        'vocab_unico': vocab_unico,
        'ricchezza_lessicale': ricchezza_lessicale,
        'aggettivi': aggettivi,
        'avverbi': avverbi,
        'verbi': verbi,
        'sostantivi': sostantivi,
        'leggibilita': leggibilita,
        'tono_positivo': tono_positivo,
        'tono_negativo': tono_negativo
    }

def determina_stile_dominante(analisi):
    """Determina lo stile di scrittura dominante"""
    punteggi = {
        'narrativo': 0,
        'descrittivo': 0,
        'argomentativo': 0,
        'espositivo': 0
    }
    
    if analisi['parole_totali'] > 0:
        # Narrativo: molti verbi
        punteggi['narrativo'] += analisi['verbi'] / analisi['parole_totali']
        punteggi['narrativo'] += 0.5 if 15 <= analisi['lunghezza_media_frasi'] <= 25 else 0
        
        # Descrittivo: molti aggettivi
        punteggi['descrittivo'] += analisi['aggettivi'] / analisi['parole_totali']
    
    # Argomentativo: ricchezza lessicale
    punteggi['argomentativo'] += analisi['ricchezza_lessicale']
    punteggi['argomentativo'] += 0.5 if analisi['lunghezza_media_frasi'] > 20 else 0
    
    # Espositivo: buona leggibilità
    punteggi['espositivo'] += 1 - abs(70 - analisi['leggibilita']) / 70
    
    return max(punteggi, key=punteggi.get)

# Strategie per intelligenze multiple
def strategie_intelligenze_multiple(stile_dominante):
    strategie = {
        'Linguistica': [
            "Amplia il tuo vocabolario leggendo autori di generi diversi",
            "Esercitati con giochi di parole e cruciverba",
            "Scrivi piccoli racconti utilizzando parole nuove",
            "Analizza la struttura di testi che ammiri"
        ],
        'Logico-Matematica': [
            "Organizza i tuoi testi con struttura logica chiara",
            "Utilizza connettivi logici per legare le idee",
            "Crea mappe concettuali prima di scrivere",
            "Supporta le argomentazioni con dati"
        ],
        'Spaziale': [
            "Usa metafore visive nelle descrizioni",
            "Disegna le scene prima di descriverle",
            "Organizza il testo con struttura visivamente chiara",
            "Utilizza diagrammi per pianificare"
        ]
    }
    
    strategie_specifiche = {
        'narrativo': [
            "Sviluppa personaggi multidimensionali",
            "Crea una struttura temporale chiara",
            "Usa dialoghi vivaci",
            "Descrivi ambienti in modo coinvolgente"
        ],
        'descrittivo': [
            "Coinvolgi tutti i sensi nelle descrizioni",
            "Usa similitudini e metafore originali",
            "Organizza le descrizioni in modo logico",
            "Scegli aggettivi precisi e evocativi"
        ],
        'argomentativo': [
            "Struttura logicamente le argomentazioni",
            "Anticipa e confuta le obiezioni",
            "Usa linguaggio preciso e convincente",
            "Supporta con esempi concreti"
        ],
        'espositivo': [
            "Organizza informazioni in modo chiaro",
            "Usa analogie per spiegare concetti complessi",
            "Adatta il linguaggio al pubblico",
            "Mantieni tono equilibrato e oggettivo"
        ]
    }
    
    return strategie, strategie_specifiche.get(stile_dominante, [])

# Interfaccia principale
tab1, tab2, tab3 = st.tabs(["📝 Analisi Testo", "📊 Risultati", "🎯 Strategie"])

with tab1:
    st.header("Inserisci il tuo testo")
    
    testo_esempio = """La città si svegliava lentamente sotto un cielo color pesca. Le prime luci del mattino accarezzavano i tetti delle case, dipingendo ombre lunghe e sottili sui marciapiedi ancora deserti. Marco camminava a passi lenti, assaporando il silenzio irreale che precede il caos della giornata. Ogni respiro gli sembrava più profondo, ogni pensiero più chiaro. In quei momenti di tranquillità, riusciva finalmente ad ascoltare la voce sottile della sua anima."""
    
    testo_utente = st.text_area(
        "Incolla il tuo testo qui sotto (minimo 50 parole):",
        value=testo_esempio,
        height=300
    )
    
    if st.button("Analizza il mio stile"):
        if nltk_available:
            parole_count = len(word_tokenize(testo_utente))
        else:
            parole_count = len(tokenizza_testo_semplice(testo_utente))
            
        if parole_count < 50:
            st.warning(f"Testo troppo breve ({parole_count} parole). Inserisci almeno 50 parole.")
        else:
            with st.spinner("Analizzando il tuo stile..."):
                analisi = analizza_stile(testo_utente)
                stile_dominante = determina_stile_dominante(analisi)
                
                st.session_state.analisi = analisi
                st.session_state.stile_dominante = stile_dominante
                
            st.success("Analisi completata!")

with tab2:
    st.header("Risultati dell'Analisi")
    
    if 'analisi' not in st.session_state:
        st.info("Inserisci un testo per vedere i risultati.")
    else:
        analisi = st.session_state.analisi
        stile_dominante = st.session_state.stile_dominante
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metriche di Base")
            metriche = {
                "Parole totali": analisi['parole_totali'],
                "Frasi totali": analisi['frasi_totali'],
                "Lunghezza media frasi": f"{analisi['lunghezza_media_frasi']:.1f} parole",
                "Vocabolario unico": analisi['vocab_unico'],
                "Ricchezza lessicale": f"{analisi['ricchezza_lessicale']*100:.1f}%",
                "Leggibilità": f"{analisi['leggibilita']:.1f}/100"
            }
            
            for k, v in metriche.items():
                st.metric(k, v)
        
        with col2:
            st.subheader("Composizione")
            df = pd.DataFrame({
                'Tipo': ['Sostantivi', 'Verbi', 'Aggettivi', 'Avverbi'],
                'Quantità': [
                    analisi['sostantivi'],
                    analisi['verbi'], 
                    analisi['aggettivi'],
                    analisi['avverbi']
                ]
            })
            st.bar_chart(df.set_index('Tipo'))
        
        st.subheader(f"Stile Dominante: {stile_dominante.capitalize()}")
        descrizioni = {
            'narrativo': "Focus su eventi e sequenze temporali",
            'descrittivo': "Focus su dettagli sensoriali e descrizioni", 
            'argomentativo': "Focus su logica e persuasione",
            'espositivo': "Focus su chiarezza e organizzazione"
        }
        st.info(descrizioni.get(stile_dominante, ""))

with tab3:
    st.header("Strategie Personalizzate")
    
    if 'analisi' not in st.session_state:
        st.info("Analizza un testo per vedere le strategie.")
    else:
        stile_dominante = st.session_state.stile_dominante
        strategie, specifiche = strategie_intelligenze_multiple(stile_dominante)
        
        st.subheader("Strategie per Inteligenze Multiple")
        for intelligenza, consigli in strategie.items():
            with st.expander(f"🧠 {intelligenza}"):
                for c in consigli:
                    st.write(f"• {c}")
        
        st.subheader(f"Strategie per Stile {stile_dominante.capitalize()}")
        for s in specifiche:
            st.write(f"• {s}")

# Footer
st.markdown("---")
st.markdown("App di Scrittura Intelligente | Sviluppa il tuo stile unico")
