# ==============================================================================
# app_cloud.py — NeuroGait (Version Optimisée - PFE)
# Système d'Analyse de la Marche Parkinsonienne
# ==============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks
import joblib
import collections
import time
import hashlib
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from fpdf import FPDF

st.set_page_config(
    page_title="NeuroGait — Analyse de Marche",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# UTILISATEURS & SÉCURITÉ
# ============================================
USERS = {
    "admin":   hashlib.sha256("admin123".encode()).hexdigest(),
    "medecin": hashlib.sha256("parkinson2026".encode()).hexdigest(),
    "lina":    hashlib.sha256("pfe2026".encode()).hexdigest(),
}

def check_password(u, p):
    return USERS.get(u) == hashlib.sha256(p.encode()).hexdigest()

# ============================================
# BASE DE DONNÉES (Sécurisée avec Context Manager)
# ============================================
def init_db():
    with sqlite3.connect("neurogait.db") as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient TEXT, age INTEGER, sexe TEXT,
            date TEXT, duree REAL,
            fog_episodes INTEGER, normal_windows INTEGER,
            score_risque INTEGER, cadence REAL,
            variabilite REAL, freeze_index REAL,
            utilisateur TEXT)""")
        conn.commit()

def save_session(patient, age, sexe, duree, n_fog, n_normal,
                 score, cadence, variabilite, fi, user):
    with sqlite3.connect("neurogait.db") as conn:
        conn.execute("""INSERT INTO sessions
            (patient,age,sexe,date,duree,fog_episodes,normal_windows,
             score_risque,cadence,variabilite,freeze_index,utilisateur)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (patient, int(age), sexe, datetime.now().strftime("%d/%m/%Y %H:%M"),
             round(duree, 1), int(n_fog), int(n_normal), int(score),
             round(cadence, 1), round(variabilite, 3), round(fi, 2), user))
        conn.commit()

def get_sessions():
    with sqlite3.connect("neurogait.db") as conn:
        return pd.read_sql("SELECT * FROM sessions ORDER BY id DESC", conn)

init_db()

# ============================================
# CSS OPTIMISÉ (Correction des contrastes)
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --navy:    #0D2137;
    --blue:    #1A5276;
    --mid:     #2E86C1;
    --light:   #AED6F1;
    --pale:    #EBF5FB;
    --text:    #1A252F;
    --muted:   #5D7A8A;
    --border:  #D6E4F0;
    --bg-grey: #F7F9FC;
}

html, body, [data-testid="stAppViewContainer"], .main {
    background: var(--bg-grey) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    color: var(--text) !important;
}

/* ── CORRECTION DES INPUTS (Contraste lisible) ── */
.main .stTextInput input, 
.main .stNumberInput input, 
.main .stSelectbox select,
.main .stSlider {
    background-color: #FFFFFF !important;
    border: 1px solid #B0C4DE !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
.main .stTextInput input::placeholder { color: #8FA3B5 !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stSelectbox select {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: white !important; 
}

/* ── CARDS & COMPOSANTS ── */
.page-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 65%, var(--mid) 100%);
    border-radius: 10px; padding: 25px 32px; margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(13,33,55,0.12); position: relative;
}
.page-header-title {
    font-family: 'Libre Baskerville', serif; font-size: 1.8rem;
    font-weight: 700; color: white; margin: 0 0 6px;
}
.page-header-sub { font-size: 0.9rem; color: rgba(255,255,255,0.75); }

.card {
    background: white; border-radius: 10px; padding: 20px;
    border: 1px solid var(--border); box-shadow: 0 1px 6px rgba(13,33,55,0.05);
}
.kpi {
    background: white; border-radius: 8px; padding: 14px 16px;
    border: 1px solid var(--border); border-top: 3px solid var(--mid);
}
.kpi.danger { border-top-color: #C0392B; }
.kpi.success { border-top-color: #27AE60; }
.kpi.warn { border-top-color: #E67E22; }

.kpi-lbl { font-size: 0.7rem; font-weight: 600; color: var(--muted); text-transform: uppercase; }
.kpi-val { font-family: 'Libre Baskerville', serif; font-size: 1.6rem; font-weight: 700; color: var(--navy); }
.kpi-note { font-size: 0.75rem; color: var(--muted); }

.score-ring {
    width: 100px; height: 100px; border-radius: 50%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    margin: 0 auto 12px; font-family: 'Libre Baskerville', serif; font-weight: 700; color: white;
}
.score-ring.low { background: linear-gradient(135deg,#1A5632,#27AE60); }
.score-ring.med { background: linear-gradient(135deg,#784212,#E67E22); }
.score-ring.high { background: linear-gradient(135deg,#641E16,#C0392B); }

.feat-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
.feat-row:last-child { border-bottom: none; }
.tag { font-size: 0.65rem; font-weight: 700; padding: 2px 8px; border-radius: 10px; text-transform: uppercase; }
.tag.ok { background: #EAFAF1; color: #1A5632; }
.tag.nok { background: #FDEDEC; color: #922B21; }
.tag.med { background: #FEF9E7; color: #784212; }

.interp-box {
    background: var(--pale); border: 1px solid var(--border); border-left: 3px solid var(--mid);
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 10px; font-size: 0.85rem;
}
.interp-box.critical { background: #FEF5F5; border-left-color: #C0392B; }
.interp-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--blue); margin-bottom: 4px; }
.interp-title.critical { color: #C0392B; }

.dx-badge { border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 14px; color: white; }
.dx-badge.fog { background: linear-gradient(135deg,#922B21,#C0392B); }
.dx-badge.normal { background: linear-gradient(135deg,#1A5632,#27AE60); }
.dx-badge.wait { background: linear-gradient(135deg,#1A252F,#2C3E50); }

.sec-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
for k, v in [('authenticated', False), ('username', ''), ('running', False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# TRAITEMENT DU SIGNAL & ML CORRIGÉS
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("models/model_fog_rf.pkl"), joblib.load("models/scaler_fog.pkl")

try:
    model, scaler = load_model()
    model_ok = True
except Exception:
    model_ok = False

def bw_filter(sig, cutoff=3.0, fs=64, order=4):
    """Filtre passe-bas pour lissage visuel et détection de foulées (cadence)."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, sig)

def bp_filter(sig, lowcut=0.5, highcut=20.0, fs=64, order=4):
    """CRITIQUE : Filtre passe-bande dédié à l'extraction ML. Préserve la bande FoG 3-8 Hz."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig)

def extract_features(win, fs=64):
    feats = []
    for ax in range(win.shape[1]):
        s = win[:, ax]
        feats += [np.mean(s), np.std(s), np.max(s), np.min(s),
                  np.max(s) - np.min(s), np.sqrt(np.mean(s**2))]
        
        # Transformation de Fourier
        fv = np.abs(np.fft.rfft(s))
        fr = np.fft.rfftfreq(len(s), d=1/fs)
        feats.append(np.sum(fv))
        
        # Calcul du Freeze Index : Puissance(3-8 Hz) / Puissance(0.5-3 Hz)
        fz = fv[(fr >= 3.0) & (fr <= 8.0)]
        lc = fv[(fr >= 0.5) & (fr <= 3.0)]
        fi = np.sum(fz**2) / (np.sum(lc**2) + 1e-6)
        feats.append(fi)
    return np.array(feats)

def predict_win(win, model, scaler):
    w = win.copy()
    # Application du filtre passe-bande correct pour l'inférence
    for i in range(3): 
        w[:, i] = bp_filter(w[:, i])
    f = extract_features(w).reshape(1, -1)
    fs = scaler.transform(f)
    pred = model.predict(fs)[0]
    prob = model.predict_proba(fs)[0][1]
    return int(pred), round(float(prob) * 100, 1)

# ============================================
# LOGIQUE MÉDICALE
# ============================================
def calc_score(cadence, variab, fi, prob_fog, n_fog, duree):
    s = 0
    if cadence < 90: s += 20
    elif cadence < 100: s += 10
    if variab > 0.5: s += 25
    elif variab > 0.2: s += 12
    if fi > 3.0: s += 20
    elif fi > 1.5: s += 10
    s += int(prob_fog * 0.20)
    if duree > 0:
        freq = (n_fog / duree) * 60
        if freq > 0.5: s += 15
        elif freq > 0.2: s += 8
    return min(int(s), 100)

def risk_level(score):
    if score < 30: return "Faible", "low", "#27AE60"
    if score < 65: return "Modéré", "med", "#E67E22"
    return "Élevé", "high", "#C0392B"

def interpret_signal(cadence, variab, fi, score, fog_pct, n_fog_real):
    anomalies = []
    if cadence < 90:
        anomalies.append(f"Cadence réduite ({cadence:.0f} ppm vs norme ~110). Signe d'une bradykinésie sévère.")
    if variab > 0.2:
        anomalies.append(f"Forte variabilité du temps de foulée ({variab:.3f} s). Instabilité posturale dynamique marquée.")
    if fi > 1.5:
        anomalies.append(f"Freeze Index critique ({fi:.2f} > seuil 1.5). Hypersynchronie spectrale dans la bande 3–8 Hz caractéristique d'un blocage moteur.")
    if fog_pct > 5:
        anomalies.append(f"Présence confirmée d'épisodes FoG ({fog_pct:.1f}% du tracé, {n_fog_real} séquences critiques).")

    if score < 30:
        concl = "Marche parkinsonienne stable et compensée. Poursuite du protocole actuel."
    elif score < 65:
        concl = "Dégradation modérée du contrôle locomoteur. Ajustement de la stimulation ou de la L-Dopa à envisager."
    else:
        concl = "Risque de chute imminent lié à des enrayements cinétiques majeurs. Évaluation neurologique prioritaire."
    return anomalies, concl

# ============================================
# RAPPORT PDF OPTIMISÉ
# ============================================
def gen_pdf(patient, age, sexe, date, duree, n_fog, n_normal, score, cadence, variab, fi, user):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    niveau, _, _ = risk_level(score)

    # En-tête
    pdf.set_fill_color(13, 33, 55)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_xy(15, 10)
    pdf.cell(0, 8, "NeuroGait — Évaluation Biomécanique", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(15, 20)
    pdf.cell(0, 6, "Analyse par capteurs inertiels (IMU) & IA — Rapport d'Examen")

    # Informations Patient
    pdf.set_xy(15, 42)
    pdf.set_text_color(13, 33, 55)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Dossier Patient", ln=True)
    pdf.set_draw_color(46, 134, 193)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    infos = [("Identifiant", patient), ("Âge / Sexe", f"{age} ans / {sexe}"), ("Date d'analyse", date), ("Opérateur", user)]
    for lbl, val in infos:
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(40, 5, f"{lbl} :", ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, str(val), ln=True)

    # Score
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Indice de Risque Global", ln=True)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    pdf.set_fill_color(230, 235, 240)
    pdf.rect(15, pdf.get_y(), 180, 8, 'F')
    fill_c = (26,86,50) if score < 30 else ((120,66,18) if score < 65 else (160,40,30))
    pdf.set_fill_color(*fill_c)
    pdf.rect(15, pdf.get_y(), int(180 * score / 100), 8, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(15, pdf.get_y() + 0.5)
    pdf.cell(180, 7, f"   {score}/100 — Risque {niveau.upper()}", ln=True)

    # Tableau des métriques
    pdf.ln(8)
    pdf.set_text_color(13, 33, 55)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Paramètres Cinématiques", ln=True)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    hdrs = ["Métrique", "Mesure", "Référence", "Interprétation"]
    cw = [60, 35, 45, 40]
    pdf.set_fill_color(13, 33, 55)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(hdrs): 
        pdf.cell(cw[i], 7, h, border=1, fill=True, align='C')
    pdf.ln()

    rows = [
        ("Durée d'enregistrement", f"{duree:.1f} s", "—", "—"),
        ("Cadence de marche", f"{cadence:.1f} ppm", "100 - 120 ppm", "Anormal" if cadence < 90 else "Normal"),
        ("Variabilité temporelle", f"{variab:.3f} s", "< 0.05 s", "Anormal" if variab > 0.2 else "Normal"),
        ("Index de gélification (FI)", f"{fi:.2f}", "< 1.5", "Critique" if fi > 1.5 else "Normal"),
        ("Séquences FoG (IA)", str(n_fog), "0", "Détecté" if n_fog > 0 else "RAS")
    ]
    pdf.set_font("Helvetica", "", 9)
    for i, row in enumerate(rows):
        pdf.set_fill_color(245, 247, 250) if i % 2 == 0 else pdf.set_fill_color(255, 255, 255)
        for j, c in enumerate(row):
            pdf.set_text_color(13, 33, 55)
            if j == 3:
                if c in ["Anormal", "Critique", "Détecté"]: pdf.set_text_color(180, 40, 30)
                elif c == "Normal": pdf.set_text_color(26, 86, 50)
            pdf.cell(cw[j], 6, c, border=1, fill=True, align='C' if j>0 else 'L')
        pdf.ln()

    # Pied de page
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "NeuroGait v2.0 — Généré automatiquement. Validation clinique requise.", ln=True, align='C')
    return bytes(pdf.output())

# ============================================
# GMAIL ALERTE
# ============================================
def send_gmail(dest, patient, n_fog, score, g_user, g_pass, pdf_b=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = g_user
        msg['To'] = dest
        niveau, _, _ = risk_level(score)
        msg['Subject'] = f"[ALERTE CLINIQUE] Détection FoG — Dossier {patient}"
        
        body = f"""L'unité d'analyse NeuroGait a identifié des anomalies locomotrices sévères.
        
Patient : {patient}
Date    : {datetime.now().strftime("%d/%m/%Y %H:%M")}
FoG     : {n_fog} séquences
Score   : {score}/100 ({niveau})

Veuillez consulter le rapport d'analyse biomécanique en pièce jointe."""
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        if pdf_b:
            p = MIMEBase('application', 'octet-stream')
            p.set_payload(pdf_b)
            encoders.encode_base64(p)
            p.add_header('Content-Disposition', f'attachment; filename="NeuroGait_{patient}.pdf"')
            msg.attach(p)
            
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
            srv.login(g_user, g_pass)
            srv.sendmail(g_user, dest, msg.as_string())
        return True, "Notification transmise au service de neurologie."
    except Exception as e:
        return False, f"Erreur d'envoi : {str(e)}"

# ============================================
# PAGE : LOGIN
# ============================================
if not st.session_state['authenticated']:
    st.markdown("""<style>[data-testid="stSidebar"]{display:none!important;}</style>""", unsafe_allow_html=True)
    _, col_log, _ = st.columns([1, 1.2, 1])
    with col_log:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;'>
            <div style='font-size:3rem;'>🔬</div>
            <h2 style='color:#0D2137;font-family:Libre Baskerville;'>NeuroGait</h2>
            <p style='color:#5D7A8A;font-size:0.9rem;'>Plateforme d'Analyse Biomécanique Pathologique</p>
        </div>""", unsafe_allow_html=True)
        
        u = st.text_input("Identifiant opérateur")
        p = st.text_input("Clé d'accès", type="password")
        if st.button("Authentification", use_container_width=True):
            if check_password(u, p):
                st.session_state['authenticated'] = True
                st.session_state['username'] = u
                st.rerun()
            else:
                st.error("Accès refusé. Vérifiez vos habilitations.")
        st.info("Comptes d'évaluation : admin / admin123 | lina / pfe2026")
    st.stop()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.markdown("### 🔬 NeuroGait\n**Unité de Diagnostic**")
    st.caption(f"Connecté : {st.session_state['username']}")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    page = st.radio("Mode d'opération", ["Analyse Fichier", "Démo Temps Réel", "Base de Sessions", "Paramètres Email"])
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Dossier Actif</div>", unsafe_allow_html=True)
    nom = st.text_input("Identifiant Patient", "PATIENT-001")
    c1, c2 = st.columns(2)
    with c1: age = st.number_input("Âge", 30, 100, 68)
    with c2: sexe = st.selectbox("Sexe", ["M", "F"])
    
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Fermer la session", use_container_width=True):
        st.session_state['authenticated'] = False
        st.rerun()

# ============================================
# ROUTAGE DES PAGES
# ============================================
st.markdown(f"""
<div class="page-header">
    <div class="page-header-title">Analyse de la Marche Parkinsonienne</div>
    <div class="page-header-sub">Module de détection cinématique du Freezing of Gait (FoG)</div>
</div>
""", unsafe_allow_html=True)

if page == "Analyse Fichier":
    uploaded = st.file_uploader("Enregistrement IMU (.txt Daphnet ou .csv)", type=["txt", "csv"])
    
    if not uploaded:
        st.info("En attente de chargement d'un tracé de signaux bruts...")
        st.stop()
        
    try:
        cols = ['time', 'ankle_x', 'ankle_y', 'ankle_z', 'thigh_x', 'thigh_y', 'thigh_z', 'trunk_x', 'trunk_y', 'trunk_z', 'label']
        df = pd.read_csv(uploaded, sep=" ", header=None, names=cols)
        df = df[df['label'] != 0].reset_index(drop=True)
    except Exception:
        st.error("Structure du fichier non conforme au standard d'acquisition.")
        st.stop()

    # Pipeline ML
    WS, OL = 128, 64
    windows_t, labels_pred, labels_real, probas = [], [], [], []
    
    for start in range(0, len(df) - WS, OL):
        end = start + WS
        win = df[['ankle_x', 'ankle_y', 'ankle_z']].iloc[start:end].values
        lbl = df['label'].iloc[start:end].mode()[0]
        if model_ok:
            pred, prob = predict_win(win, model, scaler)
            labels_pred.append(pred)
            probas.append(prob)
        labels_real.append(1 if lbl == 2 else 0)
        windows_t.append(start / 64)

    fog_pred = sum(labels_pred) if model_ok else 0
    fog_real = sum(labels_real)
    duree = len(df) / 64
    az = df['ankle_z'].values

    # Calcul des descripteurs de base
    try:
        peaks, _ = find_peaks(bw_filter(az, cutoff=3.0), distance=32)
        if len(peaks) > 2:
            diffs = np.diff(peaks) / 64
            cadence = float(60 / np.mean(diffs))
            variab = float(np.std(diffs))
        else:
            cadence, variab = 100.0, 0.1
    except Exception:
        cadence, variab = 100.0, 0.1

    # Freeze Index global sur le tracé
    fv = np.abs(np.fft.rfft(az[:1024]))
    fr = np.fft.rfftfreq(1024, d=1/64)
    fz_pow = np.sum(fv[(fr >= 3.0) & (fr <= 8.0)]**2)
    lc_pow = np.sum(fv[(fr >= 0.5) & (fr <= 3.0)]**2)
    fi_moy = float(fz_pow / (lc_pow + 1e-6))

    prob_moy = float(np.mean(probas)) if probas else 0.0
    score = calc_score(cadence, variab, fi_moy, prob_moy/100, fog_pred, duree)
    niveau, sc_cls, _ = risk_level(score)
    
    fog_pct = (fog_real / max(len(labels_real), 1)) * 100
    anomalies, concl = interpret_signal(cadence, variab, fi_moy, score, fog_pct, fog_real)

    # Rendu des KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "", "Durée", f"{duree:.0f} s", f"Fe=64Hz"),
        (k2, "success", "Allure Régulière", str(len(labels_pred)-fog_pred), "Séquences"),
        (k3, "danger", "Détections FoG", str(fog_pred), "Blocages IA"),
        (k4, "warn" if cadence<100 else "success", "Cadence", f"{cadence:.0f}", "PPM"),
        (k5, "danger" if score>=65 else "success", "Indice Risque", f"{score}/100", niveau.upper())
    ]
    for col, cls, lbl, val, nt in kpis:
        with col:
            st.markdown(f'<div class="kpi {cls}"><div class="kpi-lbl">{lbl}</div><div class="kpi-val">{val}</div><div class="kpi-note">{nt}</div></div>', unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Courbes
    c_sig, c_side = st.columns([3, 1])
    with c_sig:
        st.markdown("<div class='sec-label'>Cinématique de la cheville (Axe Z)</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bw_filter(az)[:3000], mode='lines', name='Signal lissé', line=dict(color='#1A5276', width=1.5)))
        
        # Injection discrète des surbrillances pathologiques
        in_fog = False; sf = 0
        for i in range(min(3000, len(df))):
            if df['label'].iloc[i] == 2 and not in_fog:
                sf = i; in_fog = True
            elif df['label'].iloc[i] != 2 and in_fog:
                fig.add_vrect(x0=sf, x1=i, fillcolor="#C0392B", opacity=0.15, line_width=0)
                in_fog = False
                
        fig.update_layout(height=280, margin=dict(l=5, r=5, t=5, b=5), plot_bgcolor='white')
        fig.update_xaxes(showgrid=True, gridcolor='#E8EEF3')
        fig.update_yaxes(showgrid=True, gridcolor='#E8EEF3')
        st.plotly_chart(fig, use_container_width=True)

    with c_side:
        st.markdown(f"""<div class="card" style="text-align:center;">
            <div class="score-ring {sc_cls}"><div style="font-size:2rem;">{score}</div></div>
            <b>Niveau de Risque : {niveau}</b>
        </div>""", unsafe_allow_html=True)
        
        st.markdown("<br><div class='card'><b>Marqueurs Clés</b><hr style='margin:5px 0;'>", unsafe_allow_html=True)
        st.write(f"• **FI Moy :** {fi_moy:.2f}")
        st.write(f"• **Var :** {variab:.3f} s")
        st.write(f"• **P(FoG) :** {prob_moy:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><div class='sec-label'>Synthèse Biomécanique</div>", unsafe_allow_html=True)
    for a in anomalies:
        st.markdown(f"<div class='interp-box critical'>• {a}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='interp-box'><b>Avis d'orientation :</b> {concl}</div>", unsafe_allow_html=True)

    # Actions de sortie
    st.markdown("<hr>", unsafe_allow_html=True)
    pdf_data = gen_pdf(nom, age, sexe, datetime.now().strftime("%d/%m/%Y"), duree, fog_pred, len(labels_pred)-fog_pred, score, cadence, variab, fi_moy, st.session_state['username'])
    
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        st.download_button("Exporter le Dossier (PDF)", data=pdf_data, file_name=f"NeuroGait_{nom}.pdf", mime="application/pdf", use_container_width=True)
    with col_btn2:
        if st.button("Archiver la session", use_container_width=True):
            save_session(nom, age, sexe, duree, fog_pred, len(labels_pred)-fog_pred, score, cadence, variab, fi_moy, st.session_state['username'])
            st.success("Données synchronisées dans la base locale.")

elif page == "Démo Temps Réel":
    st.info("Mode monitoring. En déploiement réel, ce module se connecte au port série du microcontrôleur (ex. ESP32 via UART/Bluetooth).")
    # Conservez votre logique visuelle d'animation ici pour la soutenance
    # ...

elif page == "Base de Sessions":
    st.markdown("<div class='sec-label'>Registre Clinique des Sessions</div>", unsafe_allow_html=True)
    df_hist = get_sessions()
    if not df_hist.empty:
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.write("Aucune session archivée actuellement.")

elif page == "Paramètres Email":
    st.markdown("<div class='sec-label'>Passerelle d'Alerte Automatique</div>", unsafe_allow_html=True)
    st.write("Configurez les accès d'envoi SMTP sécurisé (SSL/TLS).")
    
    c1, c2 = st.columns(2)
    with c1:
        g_user = st.text_input("Compte de service (Email)")
        g_pass = st.text_input("Mot de passe d'application", type="password")
    with c2:
        dest = st.text_input("Adresse de garde (Destinataire)")
        
    if st.button("Tester la passerelle", type="primary"):
        if g_user and g_pass and dest:
            # Génération d'un PDF à la volée pour le test
            pdf_dummy = gen_pdf("TEST-SYS", 65, "M", "Aujourd'hui", 60, 5, 20, 75, 95, 0.3, 2.1, "Admin")
            ok, msg = send_gmail(dest, "TEST-SYS", 5, 75, g_user, g_pass, pdf_dummy)
            if ok: st.success(msg)
            else: st.error(msg)
        else:
            st.warning("Renseignez l'ensemble des paramètres de routage.")
