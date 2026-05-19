import io
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from mistralai import Mistral
from docx import Document
import PyPDF2
import pandas as pd
import os
import time
import tempfile
from dotenv import load_dotenv
import whisper
from streamlit_mic_recorder import mic_recorder
from analyzer import analyseer_iso
from file_reader import laad_bestand

load_dotenv()

# =========================
# WHISPER SETUP
# =========================

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# =========================
# NLTK SETUP
# =========================

@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    return True

setup_nltk()

# =========================
# EMBEDDING MODEL
# =========================

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# =========================
# PROBLEM SIGNALS
# =========================

PROBLEM_SIGNALS = [
    "missing documentation", "delay or overdue", "risk or exposure",
    "unclear ownership", "miscommunication", "outdated information",
    "duplicate or redundant work", "bottleneck or blockage", "non-compliance",
    "lack of controls", "unauthorized access", "inconsistency",
    "error or mistake", "incomplete process",
]

# =========================
# THEME SETUP
# =========================

if "thema" not in st.session_state:
    st.session_state.thema = "licht"
if "chat_geschiedenis" not in st.session_state:
    st.session_state.chat_geschiedenis = {}
if "document_list" not in st.session_state:
    st.session_state.document_list = [] 
if "live_audit_log" not in st.session_state:
    st.session_state.live_audit_log = []

thema = st.session_state.thema

if thema == "festival":
    sidebar_header = "🎟️ VIP Deck / Backstage"
    api_label = "VIP Polsbandje (API Code)"
    upload_header = "🎫 Scan je tickets"
    upload_label = "Kies bestanden om te scannen"
    upload_help = "Toegestane tickets: .txt en .pdf"
    filter_header = "🎛️ DJ Mengpaneel"
    laad_tekst = "🎸 Line-up aan het samenstellen..."
    lbl_hoog = "🚙"; lbl_gem = "🥱"; lbl_laag = "💻"
    geen_bevindingen = "🎉 Geen risico's gevonden, ga maar bier halen!"
    lbl_totaal = "Festivalgangers 🕺"
    lbl_tokens = "Gedronken biertjes 🍻"
    lbl_probleem = "🔥 Moshpit Gevaar"
    lbl_aanbeveling = "🚑 EHBO-Post"
    btn_analyse = "🎧 Drop de Bass & Start Analyse!"
else:
    sidebar_header = "⚙️ Instellingen"
    api_label = "Mistral API-sleutel"
    upload_header = "📂 Documenten uploaden"
    upload_label = "Kies één of meerdere bestanden"
    upload_help = "Ondersteunde bestandstypen: .txt en .pdf"
    filter_header = "🔎 Filters"
    laad_tekst = "⏳ Mistral AI is aan het analyseren, even geduld..."
    lbl_hoog = "Hoog"; lbl_gem = "Gemiddeld"; lbl_laag = "Laag"
    geen_bevindingen = "Geen bevindingen gevonden voor de geselecteerde filters."
    lbl_totaal = "📊 Totaal"
    lbl_tokens = "Tokens"
    lbl_probleem = "Probleem"
    lbl_aanbeveling = "Aanbeveling"
    btn_analyse = "Analyseer alle documenten"

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(page_title="AI Audit Suite", page_icon="🔍", layout="wide")

# =========================
# GLOBAL STYLING
# =========================

st.markdown("""
<style>
    .stDeployButton, #MainMenu, footer, [data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebarNav"] button, button[kind="headerNoSpacing"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebar"] { min-width: 300px !important; max-width: 300px !important; }
    .doc-card {
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.9rem;
    }
    .transcription-box {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        line-height: 1.6;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

if thema == "donker":
    css = """<style>
        .stApp, [data-testid="stAppViewContainer"] { background-color: #1a1a2e !important; color: #f1f5f9 !important; }
        section[data-testid="stSidebar"] { background-color: #16213e !important; border-right: 2px solid #e879a0 !important; }
        section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
        [data-testid="stTextInput"] > div { background-color: #2a2a4a !important; border: 1px solid #e879a0 !important; border-radius: 8px !important; }
        [data-testid="stTextInput"] > div > div { background-color: transparent !important; }
        .stTextInput input, .stTextArea textarea { background-color: #2a2a4a !important; color: #f1f5f9 !important; border: none !important; }
        [data-testid="stTextInput"] button { background-color: #2a2a4a !important; border: none !important; box-shadow: none !important; }
        [data-testid="stTextInput"] button svg { fill: #f9a8d4 !important; stroke: #f9a8d4 !important; }
        [data-testid="stFileUploader"] { background-color: #2a2a4a !important; border: 1px solid #e879a0 !important; border-radius: 8px !important; }
        [data-testid="stFileUploader"] * { color: #f1f5f9 !important; }
        [data-testid="stFileUploaderDropzone"] { background-color: #2a2a4a !important; }
        [data-testid="stFileUploaderDropzone"] button, [data-testid="stFileUploader"] button { background-color: #e879a0 !important; color: white !important; border: none !important; border-radius: 8px !important; }
        .stButton > button { background-color: #e879a0 !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
        .stButton > button:hover { background-color: #be185d !important; }
        [data-testid="stMetric"] { background-color: #2a2a1a !important; border: 1px solid #a16207 !important; border-radius: 10px !important; padding: 12px !important; }
        [data-testid="stMetric"] * { color: #fef9c3 !important; }
        [data-testid="stExpander"] { border: 1px solid #e879a0 !important; border-radius: 8px !important; background-color: #1f1f3a !important; }
        h1, h2, h3 { color: #f9a8d4 !important; }
        hr { border-color: #e879a0 !important; }
        [data-testid="stChatInput"] { background-color: #2a2a4a !important; border: 1px solid #e879a0 !important; border-radius: 12px !important; }
        [data-testid="stChatInput"] textarea { background-color: transparent !important; color: #f1f5f9 !important; }
        [data-testid="stChatMessage"] { background-color: #1f1f3a !important; border: 1px solid #e879a0 !important; border-radius: 8px !important; padding: 15px !important; }
        [data-testid="stBottom"] > div { background-color: #1a1a2e !important; }
        .doc-card { background-color: #1f1f3a; border: 1px solid #e879a0; color: #f1f5f9; }
        .transcription-box { background-color: #2a2a4a; border: 1px solid #e879a0; color: #f1f5f9; }
    </style>"""
elif thema == "festival":
    css = """<style>
        @keyframes strobeLight { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        @keyframes neonFlicker { 0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% { text-shadow: 0 0 5px #fff, 0 0 20px #ff00de, 0 0 80px #ff00de; } 20%, 24%, 55% { text-shadow: none; } }
        @keyframes bassDrop { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.02); } }
        * { cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><text y='24' font-size='24'>🪩</text></svg>"), auto !important; }
        .stApp, [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #120b29, #2b1055, #4a0e4e, #120b29) !important; background-size: 300% 300% !important; animation: strobeLight 6s ease infinite !important; color: #00ffff !important; }
        section[data-testid="stSidebar"] { background: rgba(18, 11, 41, 0.8) !important; border-right: 3px solid #ff00de !important; }
        section[data-testid="stSidebar"] * { color: #00ffff !important; }
        .stTextInput input, .stTextArea textarea { background-color: rgba(0,0,0,0.8) !important; color: #ffea00 !important; border: 2px solid #00ffff !important; border-radius: 12px !important; }
        [data-testid="stFileUploader"] { background: rgba(0,0,0,0.4) !important; border: 2px dashed #ff00de !important; border-radius: 16px !important; }
        [data-testid="stFileUploader"] * { color: #00ffff !important; }
        .stButton > button { background: linear-gradient(135deg, #ffea00, #ff00de, #00ffff) !important; background-size: 200% 200% !important; animation: strobeLight 2s ease infinite !important; color: #120b29 !important; border: none !important; border-radius: 12px !important; font-weight: 800 !important; text-transform: uppercase !important; }
        [data-testid="stMetric"] { background: rgba(0,0,0,0.6) !important; border: 2px solid #00ffff !important; border-radius: 14px !important; padding: 12px !important; animation: bassDrop 2s ease-in-out infinite !important; }
        [data-testid="stMetric"] * { color: #ffea00 !important; font-weight: bold !important; }
        [data-testid="stExpander"] { background: rgba(0,0,0,0.6) !important; border: 2px solid #ff00de !important; border-radius: 14px !important; }
        h1 { color: #fff !important; animation: neonFlicker 4s infinite !important; font-weight: 900 !important; text-transform: uppercase !important; }
        h2, h3 { color: #ffea00 !important; }
        hr { border: 2px solid transparent !important; background: linear-gradient(90deg, #ff00de, #00ffff, #ffea00) !important; }
        [data-testid="stChatInput"] { background: rgba(0,0,50,0.8) !important; border: 2px solid #00ffff !important; border-radius: 12px !important; }
        [data-testid="stChatInput"] textarea { background-color: transparent !important; color: #ffea00 !important; }
        [data-testid="stChatMessage"] { background-color: rgba(0,0,0,0.4) !important; border: 1px dashed #ff00de !important; border-radius: 12px !important; padding: 15px !important; }
        .doc-card { background: rgba(0,0,0,0.5); border: 1px solid #ff00de; color: #00ffff; }
        .transcription-box { background: rgba(0,0,0,0.5); border: 1px solid #00ffff; color: #ffea00; }
    </style>"""
else:
    css = ss = """<style>
        .stApp { background-color: #f8f9fa; color: #1a1a1a; }
        section[data-testid="stSidebar"] { background-color: #005B94; border-right: 2px solid #00AEEF; }
        .stButton > button { background-color: #005B94; color: white; border: none; border-radius: 8px; font-weight: 600; }
        .stButton > button:hover { background-color: #00AEEF; color: white; }
        [data-testid="stMetric"] { background-color: #6AAA3A; border: 1px solid #00AEEF; border-radius: 10px; padding: 12px; color: white; }
        [data-testid="stExpander"] { border: 1px solid #005B94; border-radius: 8px; background-color: #f0f8ff; }
        h1, h2, h3 { color: #005B94; }
        hr { border-color: #00AEEF; }
        [data-testid="stChatInput"] { background-color: #f0f8ff !important; border: 1px solid #00AEEF !important; border-radius: 8px !important; }
        [data-testid="stChatInput"] textarea { background-color: transparent !important; color: #1a1a1a !important; }
        [data-testid="stBottom"] > div { background-color: #f8f9fa !important; }
        .doc-card { background-color: #f0f8ff; border: 1px solid #00AEEF; color: #1a1a1a; }
        .transcription-box { background-color: #f0f8ff; border: 1px solid #00AEEF; color: #1a1a1a; }
    </style>"""

st.markdown(css, unsafe_allow_html=True)
# Maak een rij met kolommen om het logo rechtsboven te plaatsen
col1, col2 = st.columns([10, 9])
with col2:
    st.image("logo.png", width=200)

if thema == "festival":
    bpm_waarde = st.session_state.get("bpm", 128)
    strobe_snelheid = (60 / bpm_waarde) * 4
    cursor_emoji = st.session_state.get("cursor_emoji", "🪩")
    dynamic_css = f"""<style>
        * {{ cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'><text y='24' font-size='24'>{cursor_emoji}</text></svg>"), auto !important; }}
        .stApp, [data-testid="stAppViewContainer"] {{ animation-duration: {strobe_snelheid}s !important; }}
        h1 {{ animation-duration: {strobe_snelheid / 1.5}s !important; }}
    </style>"""
    st.markdown(dynamic_css, unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.title("🔍 AI Audit Suite")

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header(sidebar_header)
    api_key = st.text_input(api_label, type="password", value=os.getenv("MISTRAL_API_KEY", ""), help="Haal je sleutel op via console.mistral.ai")

    st.divider()
    st.subheader("🏗️ Project Context")
    project_context = st.text_input("Beschrijf het project (Optioneel):", help="Bijv: 'Aanleg snelweg A12'. Maakt de ISO-analyse specifieker.")

    st.divider()
    st.subheader("📋 ISO Normen")
    enable_9001 = st.toggle("ISO 9001 (Kwaliteit)", value=True)
    enable_14001 = st.toggle("ISO 14001 (Milieu)", value=True)
    enable_45001 = st.toggle("ISO 45001 (Veiligheid)", value=True)

    st.divider()
    st.subheader("🎯 Risk Scanner Gevoeligheid")
    threshold = st.slider("Detectiegevoeligheid", min_value=0.20, max_value=0.60, value=0.30, step=0.05)

    st.divider()
    st.subheader("🎤 Live Audit Mode")
    enable_audit_mode = st.toggle("Schakel live auditmodus in", value=False)

    if thema == "festival":
        st.divider()
        bpm = st.slider("🎶 BPM", min_value=60, max_value=220, value=st.session_state.get("bpm", 128), step=1)
        st.session_state.bpm = bpm

    st.divider()
    st.subheader("🎨 Weergave")
    if st.button("☀️ Licht" if thema != "licht" else "🌙 Donker"):
        st.session_state.thema = "donker" if thema == "licht" else "licht"
        st.rerun()
    if st.button("🎪 Festival Thema" if thema != "festival" else "🔙 Normaal thema"):
        st.session_state.thema = "festival" if thema != "festival" else "licht"
        st.rerun()

    st.divider()
    st.caption("AI Audit Suite · v2.1")

# =========================
# FESTIVAL BANNER
# =========================

if thema == "festival":
    emoji_opties = ["🎪", "🎸", "🔊", "🪩", "🕺", "🍻", "⛺", "🎶"]
    cols = st.columns(len(emoji_opties))
    for i, emoji in enumerate(emoji_opties):
        with cols[i]:
            if st.button(emoji, key=f"banner_{emoji}"):
                st.session_state.cursor_emoji = emoji
                st.rerun()

# =========================
# HELPER FUNCTIONS
# =========================

def extract_text_from_pdf(f) -> str:
    reader = PyPDF2.PdfReader(f)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def transcribe_audio_file(file_bytes: bytes, suffix: str) -> str:
    """Save audio bytes to a temp file and transcribe with Whisper."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        result = whisper_model.transcribe(tmp_path, language="nl")
        return result["text"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def split_into_sentences(text: str):
    sentences = sent_tokenize(text, language="dutch")
    return [s.strip() for s in sentences if len(s.strip()) > 30]

def detect_problem_sentences(sentences, threshold=0.30):
    signal_embeddings = embed_model.encode(PROBLEM_SIGNALS)
    sentence_embeddings = embed_model.encode(sentences)
    similarity_matrix = cosine_similarity(sentence_embeddings, signal_embeddings)
    results = []
    for i, scores in enumerate(similarity_matrix):
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score >= threshold:
            results.append({"sentence": sentences[i], "issue_type": PROBLEM_SIGNALS[best_idx], "score": best_score})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def generate_audit_questions(client, sentence, issue_type):
    prompt = f"""You are a senior internal auditor reviewing a document.
The following sentence was flagged as potentially problematic:
"{sentence}"
Detected issue category: {issue_type}
Generate exactly 3 sharp, professional audit follow-up questions.
- Be specific to the sentence content
- Do not repeat the sentence
- Return only the 3 questions as a numbered list (1. 2. 3.)"""
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=300,
    )
    return response.choices[0].message.content.strip()

def haal_gecachete_analyse_op(api_sleutel, tekst, geselecteerde_normen):
    tijdelijke_client = Mistral(api_key=api_sleutel)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return analyseer_iso(tijdelijke_client, tekst, normen=list(geselecteerde_normen))
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                st.warning(f"⏳ Rate limit bereikt. Wacht {wait} seconden... (poging {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise e

def genereer_word_rapport(doc_name, data, context, gefilterde_bevindingen):
    doc = Document()
    doc.add_heading(f'ISO Rapport — {doc_name}', 0)
    if context:
        doc.add_heading('Project Context', level=1)
        doc.add_paragraph(context)
    doc.add_heading('Management Samenvatting', level=1)
    doc.add_paragraph(data["samenvatting"])
    doc.add_heading(f'Bevindingen ({len(gefilterde_bevindingen)})', level=1)
    for b in gefilterde_bevindingen:
        doc.add_heading(f"{b['norm']} - {b['clausule']}: {b['titel']}", level=2)
        doc.add_paragraph(f"Ernst: {b['ernst'].capitalize()}")
        doc.add_paragraph(f"Probleem: {b['beschrijving']}")
        doc.add_paragraph(f"Aanbeveling: {b['aanbeveling']}")
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# =========================
# TABS
# =========================

tab1, tab2 = st.tabs(["📄 Document- & Audioanalyse", "🎤 Live Audit Modus"])

# =========================================================
# TAB 1 — DOCUMENT UPLOAD + AUDIO FILE + ANALYSIS + CHAT
# =========================================================

with tab1:
    st.caption("Upload documenten of audiobestanden · Detecteer risico's · Genereer auditbevindingen & ISO-analyses")

    # ── Audio File Upload ──
    st.subheader("🎵 Audiobestand uploaden")
    st.caption("Upload een MP3, WAV of M4A bestand. Whisper transcribeert het automatisch.")

    audio_file = st.file_uploader(
        "Kies een audiobestand",
        type=["mp3", "wav", "m4a"],
        help="Ondersteunde formaten: MP3, WAV, M4A",
        key="audio_uploader"
    )

    if audio_file is not None:
        st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")

        if st.button("🔊 Transcribeer audiobestand"):
            suffix = "." + audio_file.name.split(".")[-1]
            with st.spinner(f"Transcriberen van '{audio_file.name}' met Whisper... (dit kan even duren)"):
                transcript = transcribe_audio_file(audio_file.read(), suffix)
                st.session_state[f"audio_transcript_{audio_file.name}"] = transcript

        transcript_key = f"audio_transcript_{audio_file.name}"
        if transcript_key in st.session_state:
            transcript = st.session_state[transcript_key]
            st.success("✅ Transcriptie klaar!")
            st.markdown("**Getranscribeerde tekst:**")
            st.markdown(
                f'<div class="transcription-box">{transcript}</div>',
                unsafe_allow_html=True
            )
            col1, col2 = st.columns(2)
            with col1:
                doc_name_audio = st.text_input(
                    "Naam voor dit document:",
                    value=audio_file.name.rsplit(".", 1)[0],
                    key=f"name_{audio_file.name}"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("➕ Voeg toe aan documentenlijst", key=f"add_{audio_file.name}"):
                    existing_names = [d["name"] for d in st.session_state.document_list]
                    name_to_use = f"🎵 {doc_name_audio}" if not doc_name_audio.startswith("🎵") else doc_name_audio
                    if name_to_use not in existing_names:
                        st.session_state.document_list.append({"name": name_to_use, "text": transcript})
                        st.success(f"✅ '{name_to_use}' toegevoegd aan documentenlijst!")
                    else:
                        st.warning(f"⚠️ '{name_to_use}' staat al in de lijst.")
                    st.rerun()

    # ── Document File Upload ──
    st.divider()
    st.subheader(upload_header)
    uploaded_files = st.file_uploader(
        upload_label,
        type=["txt", "pdf"],
        help=upload_help,
        accept_multiple_files=True,
        key="doc_uploader"
    )

    if uploaded_files:
        existing_names = [d["name"] for d in st.session_state.document_list]
        added = 0
        for uf in uploaded_files:
            if uf.name not in existing_names:
                if uf.type == "application/pdf":
                    text = extract_text_from_pdf(uf)
                else:
                    text = uf.read().decode("utf-8")
                if text.strip():
                    st.session_state.document_list.append({"name": uf.name, "text": text})
                    added += 1
        if added:
            st.success(f"✅ {added} nieuw(e) document(en) toegevoegd.")
            st.rerun()

    # ── Document List Manager ──
    st.divider()
    st.subheader("📋 Documentenlijst")

    if not st.session_state.document_list:
        st.info("Nog geen documenten toegevoegd. Upload een bestand of audiobestand hierboven.")
    else:
        st.caption(f"{len(st.session_state.document_list)} document(en) klaar voor analyse")
        to_remove = None
        for i, doc in enumerate(st.session_state.document_list):
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(
                    f'<div class="doc-card">📄 <strong>{doc["name"]}</strong> &nbsp;·&nbsp; {len(doc["text"]):,} tekens</div>',
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("🗑️", key=f"remove_{i}", help=f"Verwijder {doc['name']}"):
                    to_remove = i

        if to_remove is not None:
            st.session_state.document_list.pop(to_remove)
            st.rerun()

        if st.button("🗑️ Verwijder alle documenten", type="secondary"):
            st.session_state.document_list = []
            st.session_state.all_results = {}
            st.session_state.chat_geschiedenis = {}
            st.rerun()

    # ── Guards (flag instead of st.stop so tab2 still works) ──
    analysis_blocked = False

    if not st.session_state.document_list:
        st.info("Voeg documenten toe om de analyse te starten.")
        analysis_blocked = True

    if not api_key:
        st.warning("⚠️ Voer je Mistral API-sleutel in via de sidebar.")
        analysis_blocked = True

    if not enable_9001 and not enable_14001 and not enable_45001:
        st.warning("⚠️ Selecteer minimaal één ISO-norm in de sidebar.")
        analysis_blocked = True

    # ── Run Analysis ──
    if not analysis_blocked:
        st.divider()
        if st.button(btn_analyse, type="primary"):
            client = Mistral(api_key=api_key)
            actieve_normen = tuple(
                n for n, enabled in [("ISO 9001", enable_9001), ("ISO 14001", enable_14001), ("ISO 45001", enable_45001)]
                if enabled
            )
            all_results = {}

            for doc in st.session_state.document_list:
                doc_name = doc["name"]
                raw_text = doc["text"]

                st.markdown(f"### 🔄 Bezig met: **{doc_name}**")

                geschatte_tokens = len(raw_text) // 4
                MAX_TOKENS = 25000
                if geschatte_tokens > MAX_TOKENS:
                    st.error(f"❌ '{doc_name}' is te groot ({geschatte_tokens} tokens, max {MAX_TOKENS}). Overgeslagen.")
                    continue

                tekst_voor_analyse = f"CONTEXT:\n{project_context}\n\nDOCUMENT:\n{raw_text}" if project_context else raw_text

                # Risk Scanner
                with st.spinner(f"🔍 Risk Scanner: {doc_name}..."):
                    sentences = split_into_sentences(raw_text)
                    detected = detect_problem_sentences(sentences, threshold=threshold)

                risk_results = []
                if detected:
                    progress = st.progress(0, text="Auditsvragen genereren...")
                    for idx, item in enumerate(detected):
                        progress.progress(idx / len(detected), text=f"Vraag {idx+1}/{len(detected)} — {doc_name}")
                        for attempt in range(4):
                            try:
                                questions = generate_audit_questions(client, item["sentence"], item["issue_type"])
                                risk_results.append({**item, "questions": questions})
                                break
                            except Exception as e:
                                if "429" in str(e) and attempt < 3:
                                    time.sleep(2 ** attempt)
                                else:
                                    st.error(f"Fout bij zin {idx+1}: {e}")
                                    break
                        if idx < len(detected) - 1:
                            time.sleep(3)
                        progress.progress((idx + 1) / len(detected))
                    progress.empty()

                # ISO Analyzer
                time.sleep(10)
                with st.spinner(f"🏗️ ISO Analyse: {doc_name}..."):
                    iso_data = haal_gecachete_analyse_op(api_key, tekst_voor_analyse, actieve_normen)

                all_results[doc_name] = {
                    "raw_text": raw_text,
                    "tekst_voor_analyse": tekst_voor_analyse,
                    "sentences_count": len(sentences),
                    "risk_results": risk_results,
                    "iso_data": iso_data,
                }
                st.success(f"✅ {doc_name} klaar!")

            st.session_state.all_results = all_results
            st.session_state.chat_geschiedenis = {}
            st.rerun()

    # ── Display Results ──
    if "all_results" in st.session_state and st.session_state.all_results:
        st.divider()
        st.header("📊 Analyseresultaten")

        for doc_name, res in st.session_state.all_results.items():
            with st.expander(f"📄 {doc_name}", expanded=True):

                # Risk Scanner Results
                st.subheader("🚨 Risk Signal Scanner")
                st.caption(f"{res['sentences_count']} zinnen geanalyseerd")
                risk_results = res["risk_results"]
                if not risk_results:
                    st.warning("⚠️ Geen risicosignalen gevonden bij de huidige gevoeligheidsdrempel.")
                else:
                    st.success(f"**{len(risk_results)}** zin(nen) gemarkeerd")
                    for i, item in enumerate(risk_results, 1):
                        with st.expander(f"Vraagset {i} — {item['issue_type'].upper()} (score: {item['score']:.2f})"):
                            st.markdown("**Gemarkeerde zin:**")
                            st.markdown(f"> {item['sentence']}")
                            st.markdown("**Auditopvolgingsvragen:**")
                            st.markdown(item["questions"])

                st.divider()

                # ISO Results
                st.subheader("🏗️ ISO Civil Analyzer")
                data = res["iso_data"]
                st.info(data["samenvatting"])

                hoog = sum(1 for b in data["bevindingen"] if b["ernst"] == "hoog")
                gemiddeld = sum(1 for b in data["bevindingen"] if b["ernst"] == "gemiddeld")
                laag = sum(1 for b in data["bevindingen"] if b["ernst"] == "laag")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric(lbl_totaal, len(data["bevindingen"]))
                col2.metric(f"🔴 {lbl_hoog}", hoog)
                col3.metric(f"🟠 {lbl_gem}", gemiddeld)
                col4.metric(f"🟡 {lbl_laag}", laag)

                st.subheader(filter_header)
                col_a, col_b = st.columns(2)
                with col_a:
                    filter_ernst = st.multiselect(
                        "Filter op ernst", ["hoog", "gemiddeld", "laag"],
                        default=["hoog", "gemiddeld", "laag"],
                        key=f"ernst_{doc_name}"
                    )
                with col_b:
                    huidige_normen = list(set(b["norm"] for b in data["bevindingen"]))
                    filter_norm = st.multiselect(
                        "Filter op norm", huidige_normen,
                        default=huidige_normen,
                        key=f"norm_{doc_name}"
                    )

                gefilterd = [b for b in data["bevindingen"] if b["ernst"] in filter_ernst and b["norm"] in filter_norm]

                if not gefilterd:
                    st.warning(geen_bevindingen)
                else:
                    st.success(f"**{len(gefilterd)}** bevinding(en) gevonden")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv_data = pd.DataFrame(gefilterd).to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "💾 Download CSV", data=csv_data,
                            file_name=f"rapport_{doc_name}.csv", mime="text/csv",
                            use_container_width=True, key=f"csv_{doc_name}"
                        )
                    with col_dl2:
                        word_data = genereer_word_rapport(doc_name, data, project_context, gefilterd)
                        st.download_button(
                            "📄 Download Word", data=word_data,
                            file_name=f"rapport_{doc_name}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True, key=f"word_{doc_name}"
                        )

                    st.write("")
                    for b in gefilterd:
                        ernst_kleur = {"hoog": "🔴", "gemiddeld": "🟠", "laag": "🟡"}.get(b["ernst"], "⚪")
                        display_ernst = {"hoog": lbl_hoog, "gemiddeld": lbl_gem, "laag": lbl_laag}.get(b["ernst"], b["ernst"])
                        with st.expander(f"{ernst_kleur} [{b['norm']} | {b['clausule']}] {b['titel']}"):
                            st.markdown(f"**Ernst:** {display_ernst}")
                            st.markdown(f"**{lbl_probleem}:** {b['beschrijving']}")
                            st.markdown(f"**{lbl_aanbeveling}:** {b['aanbeveling']}")

                st.divider()

                # Chat per document
                st.subheader(f"💬 Chat over: {doc_name}")
                st.caption("Stel vragen over dit specifieke document.")

                user_avatar = "🕺" if thema == "festival" else "👤"
                ai_avatar = "🎧" if thema == "festival" else "🤖"

                if doc_name not in st.session_state.chat_geschiedenis:
                    st.session_state.chat_geschiedenis[doc_name] = []

                for msg in st.session_state.chat_geschiedenis[doc_name]:
                    avatar = user_avatar if msg["role"] == "user" else ai_avatar
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])

                if prompt := st.chat_input(f"Vraag over {doc_name}...", key=f"chat_{doc_name}"):
                    st.session_state.chat_geschiedenis[doc_name].append({"role": "user", "content": prompt})
                    with st.chat_message("user", avatar=user_avatar):
                        st.markdown(prompt)
                    with st.chat_message("assistant", avatar=ai_avatar):
                        spinner_tekst = "Track aan het mixen..." if thema == "festival" else "Mistral denkt na..."
                        with st.spinner(spinner_tekst):
                            try:
                                client = Mistral(api_key=api_key)
                                messages = [{"role": "system", "content": f"Je bent een ISO auditor assistent. Beantwoord vragen uitsluitend op basis van dit document:\n\n{res['tekst_voor_analyse']}"}]
                                messages.extend(st.session_state.chat_geschiedenis[doc_name])
                                chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
                                antwoord = chat_response.choices[0].message.content
                                st.markdown(antwoord)
                                st.session_state.chat_geschiedenis[doc_name].append({"role": "assistant", "content": antwoord})
                            except Exception as e:
                                st.error(f"Fout tijdens chatten: {e}")

# =========================================================
# TAB 2 — LIVE AUDIT MODE (mic only, real-time questions)
# =========================================================

with tab2:
    st.subheader("🎤 Live Audit Modus")
    st.caption("Neem een auditgesprek op via je microfoon. De tool luistert, transcribeert en genereert direct auditopvolgingsvragen en ISO-koppelingen.")

    # Prerequisite warnings
    if not api_key:
        st.warning("⚠️ Voer je Mistral API-sleutel in via de sidebar om live analyse te activeren.")
    if not (enable_9001 or enable_14001 or enable_45001):
        st.warning("⚠️ Selecteer minimaal één ISO-norm in de sidebar.")
    if not enable_audit_mode:
        st.info("💡 Zet 'Schakel live auditmodus in' aan in de sidebar om automatische analyse na elke opname te starten.")

    st.divider()

    # ── Mic Recorder ──
    audio_data = mic_recorder(
        start_prompt="⏺ START — klik om op te nemen",
        stop_prompt="🔴 STOP — klik om te stoppen",
        key="recorder_tab2"
    )

    if audio_data is not None and len(audio_data.get("bytes", b"")) > 0:

        # Transcribe recording
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data["bytes"])
            temp_audio_path = temp_audio.name

        with st.spinner("🎙️ Transcriberen met Whisper..."):
            result = whisper_model.transcribe(temp_audio_path, language="nl")
            st.session_state.transcribed_text = result["text"]

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        st.success("✅ Transcriptie voltooid!")
        st.markdown("**Getranscribeerde tekst:**")
        st.markdown(
            f'<div class="transcription-box">{st.session_state.transcribed_text}</div>',
            unsafe_allow_html=True
        )

        # ── Live Audit Analysis ──
        if enable_audit_mode and api_key and (enable_9001 or enable_14001 or enable_45001):
            with st.spinner("🔍 Analyseren in live auditmodus..."):
                client = Mistral(api_key=api_key)

                # 1. Summarize the answer
                summary_prompt = f'Vat het volgende antwoord van de auditee samen in 1-2 zinnen:\n"{st.session_state.transcribed_text}"'
                summary_response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.2,
                    max_tokens=150
                )
                summary = summary_response.choices[0].message.content.strip()

                # 2. Detect risk signals
                sentences = split_into_sentences(st.session_state.transcribed_text)

                if not sentences:
                  st.warning("Geen tekst gevonden in de transcriptie.")
                  risk_results = []
                else:
                 risk_results = detect_problem_sentences(sentences, threshold=0.30)

                # 3. Generate follow-up questions per risk signal
                follow_up_questions = []
                for item in risk_results:
                    questions = generate_audit_questions(client, item["sentence"], item["issue_type"])
                    follow_up_questions.append(questions)

                # 4. Link to ISO clauses
                actieve_normen = tuple(
                    n for n, enabled in [("ISO 9001", enable_9001), ("ISO 14001", enable_14001), ("ISO 45001", enable_45001)]
                    if enabled
                )
                iso_data = haal_gecachete_analyse_op(api_key, st.session_state.transcribed_text, actieve_normen)

                live_result = {
                "timestamp": time.strftime("%H:%M:%S"),
                "transcriptie": st.session_state.transcribed_text,
                "summary": summary,
                "risk_results": risk_results,
               "follow_up_questions": follow_up_questions,
             "iso_data": iso_data,
             }
            st.session_state.live_audit_results = live_result
            st.session_state.live_audit_log.append(live_result)

            # Display live results
            st.divider()
            st.subheader("🔍 Live Audit Analyse")

            st.markdown("### 📝 Samenvatting")
            st.info(st.session_state.live_audit_results["summary"])

            st.markdown("### 🚨 Risicosignalen & Opvolgingsvragen")
            if st.session_state.live_audit_results["risk_results"]:
                for i, risk in enumerate(st.session_state.live_audit_results["risk_results"], 1):
                    st.markdown(f"**{i}. {risk['issue_type'].upper()}** (score: {risk['score']:.2f})")
                    st.markdown(f"> *{risk['sentence']}*")
                    if i <= len(st.session_state.live_audit_results["follow_up_questions"]):
                        st.markdown("**Voorgestelde opvolgvragen:**")
                        st.markdown(st.session_state.live_audit_results["follow_up_questions"][i - 1])
                    st.write("")
            else:
                st.success("✅ Geen risicosignalen gedetecteerd in deze opname.")

            st.markdown("### 🏗️ Gekoppelde ISO Clausules")
            if st.session_state.live_audit_results["iso_data"]["bevindingen"]:
                for b in st.session_state.live_audit_results["iso_data"]["bevindingen"]:
                    ernst_kleur = {"hoog": "🔴", "gemiddeld": "🟠", "laag": "🟡"}.get(b["ernst"], "⚪")
                    st.markdown(f"- {ernst_kleur} **[{b['norm']} | {b['clausule']}] {b['titel']}**")
            else:
                st.info("Geen ISO-clausules gekoppeld aan deze transcriptie.")

            col_wis, col_toevoegen = st.columns(2)
            with col_wis:
                if st.button("🗑️ Wis live auditresultaten"):
                    del st.session_state.live_audit_results
                    st.rerun()
            with col_toevoegen:
                if st.button("➕ Voeg opname toe aan documentenlijst voor volledige analyse"):
                    name = "🎤 Live Opname"
                    existing_names = [d["name"] for d in st.session_state.document_list]
                    if name not in existing_names:
                        st.session_state.document_list.append({"name": name, "text": st.session_state.transcribed_text})
                        st.success("✅ Toegevoegd! Ga naar de Documentanalyse tab voor de volledige ISO-analyse.")
                    else:
                        for d in st.session_state.document_list:
                            if d["name"] == name:
                                d["text"] = st.session_state.transcribed_text
                        st.info("🔄 Bestaande opname bijgewerkt. Ga naar de Documentanalyse tab.")
                    st.rerun()
                    # ── Session Log ──
if st.session_state.live_audit_log:
    st.divider()
    st.subheader("🗂️ Sessie Log")
    st.caption(f"{len(st.session_state.live_audit_log)} opname(s) deze sessie")

    if st.button("🗑️ Wis sessie log"):
        st.session_state.live_audit_log = []
        st.rerun()

    for i, entry in enumerate(reversed(st.session_state.live_audit_log), 1):
        with st.expander(f"Opname {len(st.session_state.live_audit_log) - i + 1} — {entry['timestamp']}"):
            st.markdown(f"**Samenvatting:** {entry['summary']}")
            st.markdown(f"**Risicosignalen:** {len(entry['risk_results'])}")
            st.markdown(f"**ISO bevindingen:** {len(entry['iso_data']['bevindingen'])}")

            # Word download per entry
            doc = Document()
            doc.add_heading(f"Live Audit Opname — {entry['timestamp']}", 0)
            doc.add_heading("Transcriptie", level=1)
            doc.add_paragraph(entry["transcriptie"])
            doc.add_heading("Samenvatting", level=1)
            doc.add_paragraph(entry["summary"])
            doc.add_heading("Risicosignalen", level=1)
            for j, risk in enumerate(entry["risk_results"], 1):
                doc.add_heading(f"{j}. {risk['issue_type'].upper()} (score: {risk['score']:.2f})", level=2)
                doc.add_paragraph(risk["sentence"])
                if j <= len(entry["follow_up_questions"]):
                    doc.add_heading("Opvolgingsvragen", level=3)
                    doc.add_paragraph(entry["follow_up_questions"][j - 1])
            doc.add_heading("ISO Bevindingen", level=1)
            for b in entry["iso_data"]["bevindingen"]:
                doc.add_heading(f"{b['norm']} | {b['clausule']}: {b['titel']}", level=2)
                doc.add_paragraph(f"Ernst: {b['ernst'].capitalize()}")
                doc.add_paragraph(f"Probleem: {b['beschrijving']}")
                doc.add_paragraph(f"Aanbeveling: {b['aanbeveling']}")
            bio = io.BytesIO()
            doc.save(bio)

            st.download_button(
                "📄 Download Word rapport",
                data=bio.getvalue(),
                file_name=f"live_audit_{entry['timestamp'].replace(':', '')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_log_{i}"
            )
