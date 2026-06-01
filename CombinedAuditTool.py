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
from supabase import create_client

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_client = create_client(supabase_url, supabase_key)

# =========================
# MODELLEN
# =========================

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    return True

setup_nltk()

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# =========================
# SUPABASE FEEDBACK
# =========================

def laad_feedback_supabase():
    try:
        goede = supabase_client.table("audit_feedback")\
            .select("*").eq("positief", True)\
            .order("timestamp", desc=True).limit(20).execute()
        slechte = supabase_client.table("audit_feedback")\
            .select("*").eq("positief", False)\
            .order("timestamp", desc=True).limit(20).execute()
        return {
            "goede_vragen": [{"zin": r["zin"], "issue_type": r["issue_type"], "vragen": r["vragen"], "score": r["score"]} for r in goede.data],
            "slechte_vragen": [{"zin": r["zin"], "issue_type": r["issue_type"], "vragen": r["vragen"], "score": r["score"]} for r in slechte.data],
        }
    except Exception as e:
        st.warning(f"⚠️ Kon feedback niet laden: {e}")
        return {"goede_vragen": [], "slechte_vragen": []}

def sla_feedback_op_supabase(zin, issue_type, vragen, score, positief):
    try:
        supabase_client.table("audit_feedback").insert({
            "zin": zin, "issue_type": issue_type,
            "vragen": vragen, "score": score, "positief": positief
        }).execute()
        return True
    except Exception as e:
        st.warning(f"⚠️ Kon feedback niet opslaan: {e}")
        return False

def verwijder_feedback_supabase(zin, issue_type):
    try:
        supabase_client.table("audit_feedback")\
            .delete().eq("zin", zin).eq("issue_type", issue_type).execute()
        return True
    except Exception as e:
        st.warning(f"⚠️ Kon feedback niet verwijderen: {e}")
        return False

# =========================
# RISICOSIGNALEN
# =========================

PROBLEM_SIGNALS = [
    "missing documentation", "delay or overdue", "risk or exposure",
    "unclear ownership", "miscommunication", "outdated information",
    "duplicate or redundant work", "bottleneck or blockage", "non-compliance",
    "lack of controls", "unauthorized access", "inconsistency",
    "error or mistake", "incomplete process",
]

# =========================
# SESSION STATE
# =========================

if "chat_geschiedenis" not in st.session_state:
    st.session_state.chat_geschiedenis = {}
if "document_list" not in st.session_state:
    st.session_state.document_list = []
if "live_audit_log" not in st.session_state:
    st.session_state.live_audit_log = []
if "all_results" not in st.session_state:
    st.session_state.all_results = {}
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = laad_feedback_supabase()

# =========================
# PAGINA CONFIG & STYLING
# =========================

st.set_page_config(page_title="AI Audit Suite", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .stDeployButton, #MainMenu, footer, [data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebarNav"] button, button[kind="headerNoSpacing"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebar"] { min-width: 300px !important; max-width: 300px !important; }
    .stApp { background-color: #f8f9fa; color: #1a1a1a; }
    section[data-testid="stSidebar"] { background-color: #005B94; border-right: 2px solid #00AEEF; }
    section[data-testid="stSidebar"] * { color: white !important; }
    .stButton > button { background-color: #005B94; color: white; border: none; border-radius: 8px; font-weight: 600; }
    .stButton > button:hover { background-color: #00AEEF; color: white; }
    [data-testid="stMetric"] { background-color: #6AAA3A; border: 1px solid #00AEEF; border-radius: 10px; padding: 12px; color: white; }
    [data-testid="stExpander"] { border: 1px solid #005B94; border-radius: 8px; background-color: #f0f8ff; }
    h1, h2, h3 { color: #005B94; }
    hr { border-color: #00AEEF; }
    [data-testid="stChatInput"] { background-color: #f0f8ff !important; border: 1px solid #00AEEF !important; border-radius: 8px !important; }
    [data-testid="stChatInput"] textarea { background-color: transparent !important; color: #1a1a1a !important; }
    [data-testid="stBottom"] > div { background-color: #f8f9fa !important; }
    .doc-card {
        background-color: #f0f8ff; border: 1px solid #00AEEF; color: #1a1a1a;
        border-radius: 10px; padding: 10px 16px; margin-bottom: 8px;
        display: flex; align-items: center; justify-content: space-between; font-size: 0.9rem;
    }
    .transcription-box {
        background-color: #f0f8ff; border: 1px solid #00AEEF; color: #1a1a1a;
        border-radius: 10px; padding: 14px 18px; margin: 10px 0;
        font-size: 0.92rem; line-height: 1.6; white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

col1, col2 = st.columns([10, 9])
with col2:
    st.image("logo.png", width=200)

st.title("🔍 AI Audit Suite")

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.header("⚙️ Instellingen")
    api_key = st.text_input("Mistral API-sleutel", type="password", value=os.getenv("MISTRAL_API_KEY", ""), help="Haal je sleutel op via console.mistral.ai")

    st.divider()
    st.subheader("🏗️ Project Context")
    project_context = st.text_input("Beschrijf het project (optioneel):", help="Bijv: 'Aanleg snelweg A12'. Maakt de ISO-analyse specifieker.")

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

    st.divider()
    st.caption("AI Audit Suite · v2.1")

# =========================
# HELPER FUNCTIES
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
    if not sentences:
        return []
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
    feedback = st.session_state.feedback_store
    few_shot_blok = ""

    goede_voorbeelden = [e for e in feedback["goede_vragen"] if e["issue_type"] == issue_type][-3:]
    slechte_voorbeelden = [e for e in feedback["slechte_vragen"] if e["issue_type"] == issue_type][-2:]

    if goede_voorbeelden:
        few_shot_blok += "\n\nVOORBEELDEN VAN GOEDE VRAGEN (gebruik als inspiratie):\n"
        for ex in goede_voorbeelden:
            few_shot_blok += f'Zin: "{ex["zin"]}"\nVragen:\n{ex["vragen"]}\n\n'

    if slechte_voorbeelden:
        few_shot_blok += "\nVOORBEELDEN VAN SLECHTE VRAGEN (vermijd deze stijl):\n"
        for ex in slechte_voorbeelden:
            few_shot_blok += f'Zin: "{ex["zin"]}"\nVragen:\n{ex["vragen"]}\n\n'

    prompt = f"""You are a senior internal auditor reviewing a document.
The following sentence was flagged as potentially problematic:
"{sentence}"
Detected issue category: {issue_type}
{few_shot_blok}
Generate exactly 3 sharp, professional audit follow-up questions.
- Be specific to the sentence content
- Do not repeat the sentence
- Return only the 3 questions as a numbered list (1. 2. 3.)"""

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
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

def genereer_totaal_word_rapport(all_results, context):
    doc = Document()
    doc.add_heading("ISO Totaalrapport — Alle Documenten", 0)
    if context:
        doc.add_heading("Project Context", level=1)
        doc.add_paragraph(context)
    for doc_name, res in all_results.items():
        data = res["iso_data"]
        doc.add_heading(f"Document: {doc_name}", level=1)
        doc.add_heading("Management Samenvatting", level=2)
        doc.add_paragraph(data["samenvatting"])
        doc.add_heading(f"Bevindingen ({len(data['bevindingen'])})", level=2)
        for b in data["bevindingen"]:
            doc.add_heading(f"{b['norm']} - {b['clausule']}: {b['titel']}", level=3)
            doc.add_paragraph(f"Ernst: {b['ernst'].capitalize()}")
            doc.add_paragraph(f"Probleem: {b['beschrijving']}")
            doc.add_paragraph(f"Aanbeveling: {b['aanbeveling']}")
        doc.add_page_break()
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

def genereer_vragen_word_rapport(all_results):
    doc = Document()
    doc.add_heading("Auditopvolgingsvragen — Alle Documenten", 0)
    for doc_name, res in all_results.items():
        risk_results = res["risk_results"]
        if not risk_results:
            continue
        doc.add_heading(f"Document: {doc_name}", level=1)
        for i, item in enumerate(risk_results, 1):
            doc.add_heading(f"Vraagset {i} — {item['issue_type'].upper()} (score: {item['score']:.2f})", level=2)
            doc.add_paragraph(f"Gemarkeerde zin:\n\"{item['sentence']}\"")
            doc.add_heading("Auditopvolgingsvragen:", level=3)
            doc.add_paragraph(item["questions"])
        doc.add_page_break()
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# =========================
# TABS
# =========================

tab1, tab2, tab3 = st.tabs(["📄 Document- & Audioanalyse", "🎤 Live Audit Modus", "🧪 Feedbackbeheer"])

# =========================================================
# TAB 1
# =========================================================

with tab1:
    st.caption("Upload documenten of audiobestanden · Detecteer risico's · Genereer auditbevindingen & ISO-analyses")

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
            with st.spinner(f"Transcriberen van '{audio_file.name}' met Whisper..."):
                transcript = transcribe_audio_file(audio_file.read(), suffix)
                st.session_state[f"audio_transcript_{audio_file.name}"] = transcript

        transcript_key = f"audio_transcript_{audio_file.name}"
        if transcript_key in st.session_state:
            transcript = st.session_state[transcript_key]
            st.success("✅ Transcriptie klaar!")
            st.markdown("**Getranscribeerde tekst:**")
            st.markdown(f'<div class="transcription-box">{transcript}</div>', unsafe_allow_html=True)

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
                        st.success(f"✅ '{name_to_use}' toegevoegd!")
                    else:
                        st.warning(f"⚠️ '{name_to_use}' staat al in de lijst.")
                    st.rerun()

    st.divider()
    st.subheader("📂 Documenten uploaden")
    uploaded_files = st.file_uploader(
        "Kies één of meerdere bestanden",
        type=["txt", "pdf"],
        help="Ondersteunde bestandstypen: .txt en .pdf",
        accept_multiple_files=True,
        key="doc_uploader"
    )

    if uploaded_files:
        existing_names = [d["name"] for d in st.session_state.document_list]
        added = 0
        for uf in uploaded_files:
            if uf.name not in existing_names:
                text = extract_text_from_pdf(uf) if uf.type == "application/pdf" else uf.read().decode("utf-8")
                if text.strip():
                    st.session_state.document_list.append({"name": uf.name, "text": text})
                    added += 1
        if added:
            st.success(f"✅ {added} nieuw(e) document(en) toegevoegd.")
            st.rerun()

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

    analysis_blocked = False
    if not st.session_state.document_list:
        analysis_blocked = True
    if not api_key:
        st.warning("⚠️ Voer je Mistral API-sleutel in via de sidebar.")
        analysis_blocked = True
    if not enable_9001 and not enable_14001 and not enable_45001:
        st.warning("⚠️ Selecteer minimaal één ISO-norm in de sidebar.")
        analysis_blocked = True

    if not analysis_blocked:
        st.divider()
        if st.button("🔍 Analyseer alle documenten", type="primary"):
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
                                    time.sleep(6 ** attempt)
                                else:
                                    st.warning(f"⚠️ Zin {idx+1} overgeslagen (rate limit).")
                                    break
                        if idx < len(detected) - 1:
                            time.sleep(8)
                        progress.progress((idx + 1) / len(detected))
                    progress.empty()

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

    if st.session_state.all_results:
        st.divider()
        st.header("📊 Analyseresultaten")

        if len(st.session_state.all_results) > 1:
            st.subheader("📦 Totaalrapporten (alle documenten)")
            col_tot1, col_tot2 = st.columns(2)
            with col_tot1:
                totaal_word = genereer_totaal_word_rapport(st.session_state.all_results, project_context)
                st.download_button(
                    "📄 Download ISO Totaalrapport (Word)", data=totaal_word,
                    file_name="iso_totaalrapport.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, key="dl_totaal_iso"
                )
            with col_tot2:
                vragen_word = genereer_vragen_word_rapport(st.session_state.all_results)
                st.download_button(
                    "❓ Download Vragenrapport (Word)", data=vragen_word,
                    file_name="auditopvolgingsvragen.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, key="dl_totaal_vragen"
                )
            st.divider()

        for doc_name, res in st.session_state.all_results.items():
            with st.expander(f"📄 {doc_name}", expanded=True):

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
                            col_pos, col_neg = st.columns(2)
                            with col_pos:
                                if st.button("👍 Goede vragen", key=f"pos_{doc_name}_{i}"):
                                    if sla_feedback_op_supabase(item["sentence"], item["issue_type"], item["questions"], item["score"], True):
                                        st.session_state.feedback_store = laad_feedback_supabase()
                                        st.success("✅ Opgeslagen!")
                            with col_neg:
                                if st.button("👎 Slechte vragen", key=f"neg_{doc_name}_{i}"):
                                    if sla_feedback_op_supabase(item["sentence"], item["issue_type"], item["questions"], item["score"], False):
                                        st.session_state.feedback_store = laad_feedback_supabase()
                                        st.warning("📝 Opgeslagen!")

                st.divider()

                st.subheader("🏗️ ISO Civil Analyzer")
                data = res["iso_data"]
                st.info(data["samenvatting"])

                hoog = sum(1 for b in data["bevindingen"] if b["ernst"] == "hoog")
                gemiddeld = sum(1 for b in data["bevindingen"] if b["ernst"] == "gemiddeld")
                laag = sum(1 for b in data["bevindingen"] if b["ernst"] == "laag")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📊 Totaal", len(data["bevindingen"]))
                col2.metric("🔴 Hoog", hoog)
                col3.metric("🟠 Gemiddeld", gemiddeld)
                col4.metric("🟡 Laag", laag)

                st.subheader("🔎 Filters")
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
                    st.warning("Geen bevindingen gevonden voor de geselecteerde filters.")
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
                        with st.expander(f"{ernst_kleur} [{b['norm']} | {b['clausule']}] {b['titel']}"):
                            st.markdown(f"**Ernst:** {b['ernst'].capitalize()}")
                            st.markdown(f"**Probleem:** {b['beschrijving']}")
                            st.markdown(f"**Aanbeveling:** {b['aanbeveling']}")

                st.divider()

                st.subheader(f"💬 Chat over: {doc_name}")
                st.caption("Stel vragen over dit specifieke document.")

                if doc_name not in st.session_state.chat_geschiedenis:
                    st.session_state.chat_geschiedenis[doc_name] = []

                for msg in st.session_state.chat_geschiedenis[doc_name]:
                    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                        st.markdown(msg["content"])

                if prompt := st.chat_input(f"Vraag over {doc_name}...", key=f"chat_{doc_name}"):
                    st.session_state.chat_geschiedenis[doc_name].append({"role": "user", "content": prompt})
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(prompt)
                    with st.chat_message("assistant", avatar="🤖"):
                        with st.spinner("Mistral denkt na..."):
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
# TAB 2 — LIVE AUDIT MODUS
# =========================================================

with tab2:
    st.subheader("🎤 Live Audit Modus")
    st.caption("Neem een auditgesprek op via je microfoon. De tool luistert, transcribeert en genereert direct auditopvolgingsvragen en ISO-koppelingen.")

    if not api_key:
        st.warning("⚠️ Voer je Mistral API-sleutel in via de sidebar.")
    if not (enable_9001 or enable_14001 or enable_45001):
        st.warning("⚠️ Selecteer minimaal één ISO-norm in de sidebar.")
    if not enable_audit_mode:
        st.info("💡 Zet 'Schakel live auditmodus in' aan in de sidebar om automatische analyse na elke opname te starten.")

    st.divider()

    audio_data = mic_recorder(
        start_prompt="⏺ START — klik om op te nemen",
        stop_prompt="🔴 STOP — klik om te stoppen",
        key="recorder_tab2"
    )

    if audio_data is not None and len(audio_data.get("bytes", b"")) > 0:

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
        st.markdown(f'<div class="transcription-box">{st.session_state.transcribed_text}</div>', unsafe_allow_html=True)

        if enable_audit_mode and api_key and (enable_9001 or enable_14001 or enable_45001):
            with st.spinner("🔍 Analyseren in live auditmodus..."):
                client = Mistral(api_key=api_key)

                summary_response = client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": f'Vat het volgende antwoord van de auditee samen in 1-2 zinnen:\n"{st.session_state.transcribed_text}"'}],
                    temperature=0.2, max_tokens=150
                )
                summary = summary_response.choices[0].message.content.strip()

                sentences = split_into_sentences(st.session_state.transcribed_text)
                if not sentences:
                    st.warning("Geen tekst gevonden in de transcriptie.")
                    risk_results = []
                else:
                    risk_results = detect_problem_sentences(sentences, threshold=threshold)

                follow_up_questions = []
                for item in risk_results:
                    follow_up_questions.append(generate_audit_questions(client, item["sentence"], item["issue_type"]))

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
                if st.button("➕ Voeg opname toe aan documentenlijst"):
                    name = "🎤 Live Opname"
                    existing_names = [d["name"] for d in st.session_state.document_list]
                    if name not in existing_names:
                        st.session_state.document_list.append({"name": name, "text": st.session_state.transcribed_text})
                        st.success("✅ Toegevoegd! Ga naar de Documentanalyse tab.")
                    else:
                        for d in st.session_state.document_list:
                            if d["name"] == name:
                                d["text"] = st.session_state.transcribed_text
                        st.info("🔄 Bestaande opname bijgewerkt.")
                    st.rerun()

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
                    "📄 Download Word rapport", data=bio.getvalue(),
                    file_name=f"live_audit_{entry['timestamp'].replace(':', '')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_log_{i}"
                )

# =========================================================
# TAB 3 — FEEDBACKBEHEER
# =========================================================

with tab3:
    st.subheader("🧪 Feedbackbeheer")
    st.info(
        "Feedback wordt opgeslagen in Supabase en blijft bewaard na het herladen van de pagina. "
        "Bij de volgende analyse worden jouw beoordelingen automatisch meegestuurd aan Mistral als voorbeelden."
    )

    feedback = st.session_state.feedback_store
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 👍 Goede voorbeelden ({len(feedback['goede_vragen'])})")
        if not feedback["goede_vragen"]:
            st.info("Nog geen goede voorbeelden opgeslagen.")
        for i, entry in enumerate(reversed(feedback["goede_vragen"]), 1):
            with st.expander(f"{i}. {entry['issue_type'].upper()}"):
                st.markdown(f"**Zin:** {entry['zin']}")
                st.markdown(f"**Vragen:**\n{entry['vragen']}")
                if st.button("🗑️ Verwijder", key=f"del_goed_{i}"):
                    verwijder_feedback_supabase(entry["zin"], entry["issue_type"])
                    st.session_state.feedback_store = laad_feedback_supabase()
                    st.rerun()

    with col2:
        st.markdown(f"### 👎 Slechte voorbeelden ({len(feedback['slechte_vragen'])})")
        if not feedback["slechte_vragen"]:
            st.info("Nog geen slechte voorbeelden opgeslagen.")
        for i, entry in enumerate(reversed(feedback["slechte_vragen"]), 1):
            with st.expander(f"{i}. {entry['issue_type'].upper()}"):
                st.markdown(f"**Zin:** {entry['zin']}")
                st.markdown(f"**Vragen:**\n{entry['vragen']}")
                if st.button("🗑️ Verwijder", key=f"del_slecht_{i}"):
                    verwijder_feedback_supabase(entry["zin"], entry["issue_type"])
                    st.session_state.feedback_store = laad_feedback_supabase()
                    st.rerun()

    st.divider()
    if st.button("🗑️ Wis alle feedback", type="secondary"):
        try:
            supabase_client.table("audit_feedback").delete().neq("id", "").execute()
            st.session_state.feedback_store = {"goede_vragen": [], "slechte_vragen": []}
            st.rerun()
        except Exception as e:
            st.error(f"Fout: {e}")

    st.divider()
    st.subheader("📊 Hoe werkt de feedbackloop?")
    st.markdown("""
    1. **Analyseer een document** in de eerste tab
    2. **Beoordeel de vraagsets** met 👍 of 👎
    3. **Bij de volgende analyse** stuurt het systeem jouw beoordelingen automatisch mee als voorbeelden aan Mistral
    4. **Mistral past zijn antwoorden aan** op basis van wat jij goed of slecht vond
    5. **De feedback blijft bewaard** in Supabase, ook na het herladen van de pagina
    """)
