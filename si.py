import streamlit as st
import pandas as pd
import numpy as np
from textstat import flesch_reading_ease
import re
from collections import Counter
import nltk
import ssl

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

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        # Se punkt_tab non è disponibile, usa punkt normale
        pass

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

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

# Sidebar per informazioni
with st.sidebar:
    st.header("Informazioni")
    st.markdown("""
    Questa app analizza:
    - **Vocabolario e complessità linguistica**
    - **Struttura delle frasi e paragrafi**
    - **Tono e stile narrativo**
    - **Coerenza e coesione testuale**
    
    Basandoti sui risultati, riceverai strategie personalizzate per migliorare 
    la tua scrittura sviluppando diverse intelligenze.
    """)

# Funzione per calcolare la diversità lessicale
def calcola_diversita_lessicale(testo):
    """Calcola la diversità lessicale come rapporto tra parole uniche e totali"""
    # Usa una tokenizzazione semplice se NLTK ha problemi
    try:
        parole = word_tokenize(re.sub(r'[^\w\s]', '', testo.lower()))
    except:
        # Fallback: tokenizzazione semplice
        parole = re.findall(r'\b\w+\b', testo.lower())
    
    if len(parole) == 0:
        return 0
    return len(set(parole)) / len(parole)

# Funzione di tokenizzazione robusta
def tokenizza_testo(testo):
    """Tokenizza il testo con fallback se NLTK ha problemi"""
    try:
        return word_tokenize(testo)
    except:
        # Fallback: tokenizzazione semplice con regex
        return re.findall(r'\b\w+\b', testo)

def conta_frasi(testo):
    """Conta le frasi con fallback se NLTK ha problemi"""
    try:
        return len(sent_tokenize(testo))
    except:
        # Fallback: conta frasi basate su punteggiatura
        return len([f for f in re.split(r'[.!?]+', testo) if f.strip()])

def analizza_pos(parole):
    """Analizza le parti del discorso con fallback"""
    try:
        tagged = pos_tag(parole)
        aggettivi = len([word for word, pos in tagged if pos in ['JJ', 'JJR', 'JJS']])
        avverbi = len([word for word, pos in tagged if pos in ['RB', 'RBR', 'RBS']])
        verbi = len([word for word, pos in tagged if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])
        sostantivi = len([word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']])
        return aggettivi, avverbi, verbi, sostantivi
    except:
        # Fallback: stima approssimativa
        return len(parole) // 10, len(parole) // 20, len(parole) // 5, len(parole) // 3

# Funzioni di analisi del testo
def analizza_stile(testo):
    """Analizza vari aspetti dello stile di scrittura"""
    
    # Pulizia del testo
    testo_pulito = re.sub(r'[^\w\s]', '', testo)
    
    # Metriche di base con fallback
    parole = tokenizza_testo(testo_pulito)
    frasi_totali = conta_frasi(testo)
    parole_totali = len(parole)
    
    # Lunghezza media delle frasi
    lunghezza_media_frasi = parole_totali / frasi_totali if frasi_totali > 0 else 0
    
    # Vocabolario unico
    vocab_unico = len(set(parole))
    ricchezza_lessicale = calcola_diversita_lessicale(testo)
    
    # Analisi POS (Part-of-Speech)
    aggettivi, avverbi, verbi, sostantivi = analizza_pos(parole)
    
    # Leggibilità
    try:
        leggibilita = flesch_reading_ease(testo)
    except:
        leggibilita = 60  # Valore di default
    
    # Complessità sintattica (parole per frase)
    complessita_sintattica = lunghezza_media_frasi
    
    # Tono (analisi semplificata)
    parole_pos_lista = ['buono', 'bello', 'fantastico', 'eccellente', 'meraviglioso', 'positivo', 'felice', 'gioia', 'bene', 'perfetto']
    parole_neg_lista = ['cattivo', 'brutto', 'terribile', 'orribile', 'pessimo', 'negativo', 'triste', 'dolore', 'male', 'difettoso']
    
    parole_positive = sum(1 for word in parole if word.lower() in parole_pos_lista)
    parole_negative = sum(1 for word in parole if word.lower() in parole_neg_lista)
    tono_positivo = parole_positive / len(parole) if parole_totali > 0 else 0
    tono_negativo = parole_negative / len(parole) if parole_totali > 0 else 0
    
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
        'complessita_sintattica': complessita_sintattica,
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
    
    # Narrativo: molti verbi, lunghezza frase media
    if analisi['parole_totali'] > 0:
        punteggi['narrativo'] += analisi['verbi'] / analisi['parole_totali']
        punteggi['narrativo'] += 0.5 if 15 <= analisi['lunghezza_media_frasi'] <= 25 else 0
    
    # Descrittivo: molti aggettivi
    if analisi['parole_totali'] > 0:
        punteggi['descrittivo'] += analisi['aggettivi'] / analisi['parole_totali']
    
    # Argomentativo: ricchezza lessicale, frasi più lunghe
    punteggi['argomentativo'] += analisi['ricchezza_lessicale']
    punteggi['argomentativo'] += 0.5 if analisi['lunghezza_media_frasi'] > 20 else 0
    
    # Espositivo: equilibrio, buona leggibilità
    punteggi['espositivo'] += 1 - abs(70 - analisi['leggibilita']) / 70
    
    return max(punteggi, key=punteggi.get)

# Funzioni per le strategie basate su intelligenze multiple
def strategie_intelligenze_multiple(stile_dominante, analisi):
    """Genera strategie personalizzate basate sulle intelligenze multiple"""
    
    strategie = {}
    
    # Intelligenza Linguistica
    strategie['Linguistica'] = [
        "Amplia il tuo vocabolario leggendo autori di generi diversi",
        "Esercitati con giochi di parole e cruciverba",
        "Scrivi piccoli racconti utilizzando parole nuove ogni giorno",
        "Analizza la struttura di testi che ammiri"
    ]
    
    # Intelligenza Logico-Matematica
    strategie['Logico-Matematica'] = [
        "Organizza i tuoi testi con una struttura logica chiara (introduzione, sviluppo, conclusione)",
        "Utilizza connettivi logici per legare le idee (pertanto, di conseguenza, inoltre)",
        "Crea mappe concettuali prima di scrivere",
        "Supporta le tue argomentazioni con dati e statistiche"
    ]
    
    # Intelligenza Spaziale
    strategie['Spaziale'] = [
        "Usa metafore visive nelle tue descrizioni",
        "Disegna le scene prima di descriverle",
        "Organizza il testo con una struttura visivamente chiara",
        "Utilizza diagrammi per pianificare la struttura del testo"
    ]
    
    # Intelligenza Corporeo-Cinestetica
    strategie['Corporeo-Cinestetica'] = [
        "Scrivi stando in piedi o camminando per stimolare la creatività",
        "Drammatizza le scene che vuoi descrivere",
        "Usa un linguaggio che coinvolga i sensi e il movimento",
        "Prendi brevi pause fisiche durante la scrittura"
    ]
    
    # Intelligenza Musicale
    strategie['Musicale'] = [
        "Leggi ad alta voce ciò che scrivi per verificarne il ritmo",
        "Ascolta musica mentre scrivi per ispirare diversi stati d'animo",
        "Usa l'allitterazione e l'assonanza per creare effetti sonori",
        "Presta attenzione al ritmo delle tue frasi"
    ]
    
    # Intelligenza Interpersonale
    strategie['Interpersonale'] = [
        "Scrivi pensando al tuo lettore ideale",
        "Chiedi feedback e impara a incorporare i suggerimenti",
        "Partecipa a gruppi di scrittura creativa",
        "Sviluppa dialoghi realistici tra i personaggi"
    ]
    
    # Intelligenza Intrapersonale
    strategie['Intrapersonale'] = [
        "Tieni un diario personale per esplorare le tue emozioni",
        "Rifletti sul tuo processo di scrittura e sui tuoi progressi",
        "Stabilisci obiettivi di scrittura personali",
        "Scrivi di esperienze personali per sviluppare autenticità"
    ]
    
    # Intelligenza Naturalistica
    strategie['Naturalistica'] = [
        "Osserva e descri dettagliatamente ambienti naturali",
        "Usa metafore tratte dal mondo naturale",
        "Studia la struttura di piante e animali per ispirare la struttura del testo",
        "Scrivi all'aperto per stimolare la creatività"
    ]
    
    # Strategie specifiche per stile dominante
    strategie_specifiche = {
        'narrativo': [
            "Sviluppa personaggi multidimensionali (Interpersonale)",
            "Crea una struttura temporale chiara (Logico-Matematica)",
            "Usa dialoghi vivaci (Linguistica)",
            "Descrivi ambienti in modo coinvolgente (Spaziale)"
        ],
        'descrittivo': [
            "Coinvolgi tutti i sensi nelle descrizioni (Corporeo-Cinestetica)",
            "Usa similitudini e metafore originali (Spaziale)",
            "Organizza le descrizioni in modo logico (Logico-Matematica)",
            "Scegli aggettivi precisi e evocativi (Linguistica)"
        ],
        'argomentativo': [
            "Struttura logicamente le argomentazioni (Logico-Matematica)",
            "Anticipa e confuta le obiezioni (Interpersonale)",
            "Usa un linguaggio preciso e convincente (Linguistica)",
            "Supporta con esempi concreti (Naturalistica)"
        ],
        'espositivo': [
            "Organizza le informazioni in modo chiaro (Logico-Matematica)",
            "Usa analogie per spiegare concetti complessi (Spaziale)",
            "Adatta il linguaggio al pubblico (Interpersonale)",
            "Mantieni un tono equilibrato e oggettivo (Intrapersonale)"
        ]
    }
    
    return strategie, strategie_specifiche.get(stile_dominante, [])

# Interfaccia principale
tab1, tab2, tab3 = st.tabs(["📝 Analisi Testo", "📊 Risultati Dettagliati", "🎯 Strategie Personalizzate"])

with tab1:
    st.header("Inserisci il tuo testo")
    
    testo_esempio = """La città si svegliava lentamente sotto un cielo color pesca. Le prime luci del mattino accarezzavano i tetti delle case, dipingendo ombre lunghe e sottili sui marciapiedi ancora deserti. Marco camminava a passi lenti, assaporando il silenzio irreale che precede il caos della giornata. Ogni respiro gli sembrava più profondo, ogni pensiero più chiaro. In quei momenti di tranquillità, riusciva finalmente ad ascoltare la voce sottile della sua anima, quella che durante il giorno veniva sommersa dalle richieste, dalle scadenze, dalle aspettative degli altri."""
    
    testo_utente = st.text_area(
        "Incolla il tuo testo qui sotto (minimo 50 parole per un'analisi accurata):",
        value=testo_esempio,
        height=300
    )
    
    if st.button("Analizza il mio stile"):
        parole_count = len(tokenizza_testo(testo_utente))
        if parole_count < 50:
            st.warning(f"Inserisci un testo più lungo (almeno 50 parole, attuali: {parole_count}) per un'analisi accurata.")
        else:
            with st.spinner("Analizzando il tuo stile di scrittura..."):
                analisi = analizza_stile(testo_utente)
                stile_dominante = determina_stile_dominante(analisi)
                
                # Salva i risultati nella session state
                st.session_state.analisi = analisi
                st.session_state.stile_dominante = stile_dominante
                
            st.success("Analisi completata! Vai alla tab 'Risultati Dettagliati'.")

with tab2:
    st.header("Risultati dell'Analisi")
    
    if 'analisi' not in st.session_state:
        st.info("Inserisci un testo nella tab 'Analisi Testo' per vedere i risultati.")
    else:
        analisi = st.session_state.analisi
        stile_dominante = st.session_state.stile_dominante
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metriche di Base")
            
            metriche_base = {
                "Parole totali": analisi['parole_totali'],
                "Frasi totali": analisi['frasi_totali'],
                "Lunghezza media frasi": f"{analisi['lunghezza_media_frasi']:.1f} parole",
                "Vocabolario unico": analisi['vocab_unico'],
                "Ricchezza lessicale": f"{analisi['ricchezza_lessicale']*100:.1f}%",
                "Livello di leggibilità": f"{analisi['leggibilita']:.1f} (su 100)"
            }
            
            for metrica, valore in metriche_base.items():
                st.metric(metrica, valore)
        
        with col2:
            st.subheader("Composizione del Testo")
            
            # Calcolo delle percentuali per le parti del discorso
            labels = ['Sostantivi', 'Verbi', 'Aggettivi', 'Avverbi', 'Altro']
            sizes = [
                analisi['sostantivi'],
                analisi['verbi'],
                analisi['aggettivi'],
                analisi['avverbi'],
                analisi['parole_totali'] - (analisi['sostantivi'] + analisi['verbi'] + analisi['aggettivi'] + analisi['avverbi'])
            ]
            
            # Creazione di un DataFrame per visualizzare i dati
            df_composizione = pd.DataFrame({
                'Parte del Discorso': labels,
                'Conteggio': sizes,
                'Percentuale': [f"{(size/analisi['parole_totali'])*100:.1f}%" for size in sizes]
            })
            
            # Visualizzazione come tabella
            st.dataframe(df_composizione, use_container_width=True)
            
            # Visualizzazione alternativa come bar chart
            st.bar_chart(df_composizione.set_index('Parte del Discorso')['Conteggio'])
        
        # Stile dominante
        st.subheader("Stile di Scrittura Dominante")
        
        stili = {
            'narrativo': "📖 Narrativo",
            'descrittivo': "🎨 Descrittivo", 
            'argomentativo': "💭 Argomentativo",
            'espositivo': "📚 Espositivo"
        }
        
        st.markdown(f"**Il tuo stile dominante è:** {stili[stile_dominante]}")
        
        descrizioni_stili = {
            'narrativo': "Lo stile narrativo si concentra sul racconto di eventi, spesso in sequenza temporale, con attenzione ai personaggi e alle loro azioni.",
            'descrittivo': "Lo stile descrittivo si focalizza sulla rappresentazione dettagliata di persone, luoghi o oggetti, utilizzando molti aggettivi e avverbi.",
            'argomentativo': "Lo stile argomentativo presenta tesi supportate da ragionamenti logici, dati e prove, con l'obiettivo di persuadere il lettore.",
            'espositivo': "Lo stile espositivo fornisce informazioni in modo chiaro e organizzato, spiegando concetti senza necessariamente persuadere."
        }
        
        st.info(descrizioni_stili[stile_dominante])

with tab3:
    st.header("Strategie Personalizzate")
    
    if 'analisi' not in st.session_state:
        st.info("Inserisci un testo nella tab 'Analisi Testo' per ricevere strategie personalizzate.")
    else:
        analisi = st.session_state.analisi
        stile_dominante = st.session_state.stile_dominante
        
        strategie, strategie_specifiche = strategie_intelligenze_multiple(stile_dominante, analisi)
        
        st.subheader("Strategie per Sviluppare Tutte le Intelligenze")
        
        for intelligenza, consigli in strategie.items():
            with st.expander(f"🧠 {intelligenza}"):
                for consiglio in consigli:
                    st.write(f"• {consiglio}")
        
        st.subheader(f"Strategie Specifiche per lo Stile {stile_dominante.capitalize()}")
        
        for strategia in strategie_specifiche:
            st.write(f"• {strategia}")
        
        # Consigli personalizzati basati sull'analisi
        st.subheader("Consigli Personalizzati")
        
        if analisi['ricchezza_lessicale'] < 0.5:
            st.warning("**Amplia il tuo vocabolario:** La ricchezza lessicale è sotto la media. Prova a leggere più generi diversi e a utilizzare un dizionario dei sinonimi.")
        
        if analisi['lunghezza_media_frasi'] > 25:
            st.warning("**Varia la lunghezza delle frasi:** Le tue frasi sono piuttosto lunghe. Prova ad alternare frasi brevi e lunghe per creare ritmo.")
        
        if analisi['aggettivi'] / analisi['parole_totali'] < 0.05:
            st.warning("**Aggiungi più dettagli descrittivi:** Usa più aggettivi per rendere la tua scrittura più vivida e coinvolgente.")
        
        if analisi['leggibilita'] < 60:
            st.warning("**Migliora la leggibilità:** Il tuo testo potrebbe essere difficile da leggere per alcuni. Prova a semplificare frasi complesse e usa un linguaggio più diretto.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>App di Scrittura Intelligente - Sviluppa il tuo stile unico attraverso le intelligenze multiple</p>
    </div>
    """,
    unsafe_allow_html=True
)
