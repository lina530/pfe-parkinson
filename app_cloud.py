# ============================================
# app_cloud.py — NeuroGait
# Version corrigée — Remarques jury PFE
# Corrections :
#   1. Filtre passe-bande 0.5–20 Hz (préserve bande FoG 3–8 Hz)
#   2. Simulation réaliste via données Daphnet réelles
#   3. SQLite avec gestionnaire de contexte
#   4. UI accessible — contrastes corrects
# ============================================
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
import os
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
# UTILISATEURS
# ============================================
USERS = {
    "admin":   hashlib.sha256("admin123".encode()).hexdigest(),
    "medecin": hashlib.sha256("parkinson2026".encode()).hexdigest(),
    "lina":    hashlib.sha256("pfe2026".encode()).hexdigest(),
}
def check_password(u, p):
    return USERS.get(u) == hashlib.sha256(p.encode()).hexdigest()

# ============================================
# BASE DE DONNÉES — Gestionnaire de contexte
# ============================================
DB_PATH = "neurogait.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                patient       TEXT,
                age           INTEGER,
                sexe          TEXT,
                date          TEXT,
                duree         REAL,
                fog_episodes  INTEGER,
                normal_windows INTEGER,
                score_risque  INTEGER,
                cadence       REAL,
                variabilite   REAL,
                freeze_index  REAL,
                utilisateur   TEXT
            )
        """)

def save_session(patient, age, sexe, duree, n_fog, n_normal,
                 score, cadence, variab, fi, user):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sessions
            (patient,age,sexe,date,duree,fog_episodes,
             normal_windows,score_risque,cadence,
             variabilite,freeze_index,utilisateur)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (patient, age, sexe,
              datetime.now().strftime("%d/%m/%Y %H:%M"),
              round(duree,1), n_fog, n_normal, score,
              round(cadence,1), round(variab,4),
              round(fi,3), user))

def get_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            "SELECT * FROM sessions ORDER BY id DESC", conn
        )

init_db()

# ============================================
# CSS — Médical bleu/blanc, contrastes corrects
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0D2137;
    --blue:   #1A5276;
    --mid:    #2E86C1;
    --light:  #AED6F1;
    --pale:   #EBF5FB;
    --white:  #FFFFFF;
    --grey:   #F7F9FC;
    --border: #D6E4F0;
    --text:   #1A252F;
    --muted:  #5D7A8A;
    --danger: #922B21;
    --ok:     #1A5632;
    --warn:   #784212;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--grey) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    color: var(--text) !important;
}

/* Réduire padding bloc principal */
.block-container { padding-top: 1rem !important; }

/* ── SIDEBAR ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }

/* Inputs sidebar */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

/* ── INPUTS PAGE PRINCIPALE ──────────────── */
/* Fond blanc, texte sombre — correction contraste jury */
.main input[type="text"],
.main input[type="password"],
.main input[type="number"],
.main textarea,
[data-testid="stTextInput"] input,
[data-testid="stPasswordInput"] input,
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stPasswordInput"] input::placeholder {
    color: #95A5A6 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stPasswordInput"] input:focus {
    border-color: var(--mid) !important;
    box-shadow: 0 0 0 3px rgba(46,134,193,0.12) !important;
}

/* Selectbox page principale */
[data-baseweb="select"] > div {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}

/* ── PAGE HEADER ─────────────────────────── */
.page-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 65%, var(--mid) 100%);
    border-radius: 10px; padding: 28px 34px;
    margin-bottom: 22px;
    box-shadow: 0 2px 14px rgba(13,33,55,0.16);
    position: relative; overflow: hidden;
}
.page-header::after {
    content: ''; position: absolute;
    right: -28px; top: -28px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
    pointer-events: none;
}
.ph-eyebrow {
    font-size: 0.67rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--light); margin-bottom: 7px;
}
.ph-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.85rem; font-weight: 700;
    color: white; letter-spacing: -0.3px; line-height: 1.15;
}
.ph-sub {
    font-size: 0.86rem; color: rgba(255,255,255,0.58);
    margin-top: 5px; font-weight: 300;
}
.ph-user {
    position: absolute; top: 18px; right: 22px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 18px; padding: 4px 13px;
    font-size: 0.78rem; color: rgba(255,255,255,0.8);
    font-weight: 500;
}

/* ── KPI ─────────────────────────────────── */
.kpi {
    background: white; border-radius: 8px;
    padding: 15px 17px;
    box-shadow: 0 1px 5px rgba(13,33,55,0.06);
    border: 1px solid var(--border);
    border-top: 3px solid var(--mid);
}
.kpi.dk  { border-top-color: var(--danger); }
.kpi.ok  { border-top-color: var(--ok); }
.kpi.wk  { border-top-color: #E67E22; }
.kpi-lbl { font-size: 0.67rem; font-weight: 700; color: var(--muted);
           text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.kpi-val { font-family: 'Libre Baskerville', serif;
           font-size: 1.7rem; font-weight: 700; color: var(--navy); line-height: 1.1; }
.kpi-note { font-size: 0.73rem; color: var(--muted); margin-top: 2px; }

/* ── CARD ────────────────────────────────── */
.card {
    background: white; border-radius: 9px;
    padding: 18px 20px;
    box-shadow: 0 1px 7px rgba(13,33,55,0.06);
    border: 1px solid var(--border);
    margin-bottom: 14px;
}
.card-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 0.9rem; font-weight: 700;
    color: var(--navy); margin-bottom: 12px;
    padding-bottom: 9px; border-bottom: 1px solid var(--border);
}

/* ── SCORE ───────────────────────────────── */
.score-ring {
    width: 105px; height: 105px; border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin: 0 auto 10px;
    font-family: 'Libre Baskerville', serif;
    font-weight: 700; color: white;
    box-shadow: 0 3px 14px rgba(0,0,0,0.17);
}
.score-ring.low  { background: linear-gradient(135deg,#1A5632,#27AE60); }
.score-ring.med  { background: linear-gradient(135deg,#784212,#E67E22); }
.score-ring.high { background: linear-gradient(135deg,#641E16,#C0392B);
                   animation: ring-pulse 2s infinite; }
@keyframes ring-pulse {
    0%,100% { box-shadow: 0 3px 14px rgba(192,57,43,.28); }
    50%      { box-shadow: 0 3px 26px rgba(192,57,43,.62); }
}
.score-num    { font-size: 1.75rem; line-height: 1; }
.score-denom  { font-size: 0.67rem; opacity: .75; }
.score-title  { font-family: 'Libre Baskerville', serif; font-size: 0.92rem;
                font-weight: 700; color: var(--navy); text-align: center; }
.score-sub    { font-size: 0.75rem; color: var(--muted); text-align: center; }

/* ── FEATURE ROW ─────────────────────────── */
.feat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
}
.feat-row:last-child { border-bottom: none; }
.feat-name { color: var(--text); font-weight: 500; }
.feat-val  { font-weight: 700;
             font-family: 'Libre Baskerville', serif; font-size: 0.92rem; }
.feat-ref  { font-size: 0.69rem; color: var(--muted); }
.tag {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    padding: 2px 7px; border-radius: 9px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.tag.ok  { background: #EAFAF1; color: #1A5632; }
.tag.nok { background: #FDEDEC; color: #922B21; }
.tag.med { background: #FEF9E7; color: #784212; }

/* ── INTERPRÉTATION ──────────────────────── */
.interp-box {
    background: var(--pale); border: 1px solid var(--border);
    border-left: 3px solid var(--mid);
    border-radius: 0 8px 8px 0;
    padding: 13px 17px; margin-bottom: 9px;
    font-size: 0.84rem; color: var(--text); line-height: 1.65;
}
.interp-box.crit { background: #FEF5F5; border-left-color: var(--danger); }
.interp-title { font-size: 0.67rem; font-weight: 700;
                text-transform: uppercase; letter-spacing: 1px;
                color: var(--blue); margin-bottom: 4px; }
.interp-title.crit { color: var(--danger); }

/* ── DIAGNOSTIC ──────────────────────────── */
.dx {
    border-radius: 8px; padding: 16px 18px;
    text-align: center; margin-bottom: 12px;
}
.dx.fog    { background: linear-gradient(135deg,#922B21,#C0392B);
             animation: ring-pulse 2s infinite; }
.dx.normal { background: linear-gradient(135deg,#1A5632,#27AE60); }
.dx.wait   { background: linear-gradient(135deg,#1A252F,#2C3E50); }
.dx-icon   { font-size: 1.8rem; margin-bottom: 5px; }
.dx-title  { font-family: 'Libre Baskerville', serif;
             font-size: 1.1rem; font-weight: 700; color: white; }
.dx-sub    { font-size: .77rem; color: rgba(255,255,255,.7); margin-top: 2px; }

/* ── PROBA BAR ───────────────────────────── */
.pb-wrap { background: white; border-radius: 7px;
           padding: 13px 15px; border: 1px solid var(--border);
           margin-bottom: 11px; }
.pb-lbl  { font-size: .67rem; font-weight: 700; color: var(--muted);
           text-transform: uppercase; letter-spacing: 1px; margin-bottom: 7px; }
.pb-bg   { background: #E8EDF2; border-radius: 4px; height: 7px; overflow: hidden; }
.pb-fill { height: 100%; border-radius: 4px; transition: width .35s ease; }
.pb-num  { font-family: 'Libre Baskerville', serif;
           font-size: 1.35rem; font-weight: 700; margin-top: 4px; }

/* ── ALERT ───────────────────────────────── */
.alert-crit {
    background: linear-gradient(90deg,#641E16,#922B21);
    border-radius: 7px; padding: 13px 17px; color: white; margin-bottom: 12px;
}
.alert-crit b { font-weight: 700; font-size: .88rem; display: block; }
.alert-crit small { font-size: .77rem; opacity: .82; }

/* ── SECTION LABEL ───────────────────────── */
.sec-lbl {
    font-size: .66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: var(--muted); margin-bottom: 12px;
    border-bottom: 1px solid var(--border); padding-bottom: 5px;
}

/* ── BUTTONS ─────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg,var(--navy),var(--blue)) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: .86rem !important;
    padding: 8px 18px !important; transition: opacity .18s !important;
}
.stButton > button:hover { opacity: .80 !important; }
.stDownloadButton > button {
    background: white !important; color: var(--navy) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: .86rem !important;
}

/* ── HIDE STREAMLIT DEFAULT UI ───────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton,
[data-testid="stToolbar"] { display: none; }
hr { border-color: rgba(255,255,255,.08) !important; }

/* ── FILE UPLOADER ───────────────────────── */
[data-testid="stFileUploader"] {
    background: white; border-radius: 8px;
    border: 1.5px dashed var(--border) !important;
    padding: 10px;
}

/* ── DATAFRAME ───────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
for k, v in [('authenticated', False), ('username', ''),
             ('running', False), ('demo_idx', 0),
             ('demo_df', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# ML — Chargement modèle
# ============================================
@st.cache_resource
def load_model():
    m = joblib.load("models/model_fog_rf.pkl")
    s = joblib.load("models/scaler_fog.pkl")
    return m, s

try:
    model, scaler = load_model()
    model_ok = True
except Exception:
    model_ok = False

# ============================================
# CORRECTION 1 — Filtre PASSE-BANDE 0.5–20 Hz
# Préserve la bande FoG (3–8 Hz) et la bande
# locomotrice (0.5–3 Hz). Élimine :
#   – dérive basse fréquence (gravité, < 0.5 Hz)
#   – bruit électronique haute fréquence (> 20 Hz)
# ============================================
def bandpass_filter(signal: np.ndarray,
                    low: float = 0.5,
                    high: float = 20.0,
                    fs: int = 64,
                    order: int = 4) -> np.ndarray:
    """
    Filtre passe-bande Butterworth.
    Contrairement au passe-bas à 3 Hz (erreur initiale),
    ce filtre préserve intactes les deux bandes d'intérêt :
      · Locomotion : 0.5–3 Hz
      · Freeze     : 3–8 Hz  ← signature spectrale du FoG
    """
    nyq = 0.5 * fs
    low_n  = low  / nyq
    high_n = high / nyq
    # Clamp pour éviter erreur numérique
    low_n  = max(low_n,  0.001)
    high_n = min(high_n, 0.999)
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal)

# Filtre passe-bas léger uniquement pour visualisation
def lowpass_viz(signal, cutoff=10, fs=64, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, min(cutoff/nyq, 0.999), btype='low')
    return filtfilt(b, a, signal)

def extract_features(window: np.ndarray, fs: int = 64) -> np.ndarray:
    """
    Extraction des features sur signal filtré passe-bande.
    Le Freeze Index est maintenant calculé sur un signal
    qui contient réellement de l'énergie dans 3–8 Hz.
    """
    features = []
    for ax in range(window.shape[1]):
        s = window[:, ax]
        # Features temporelles
        features += [
            np.mean(s), np.std(s),
            np.max(s),  np.min(s),
            np.max(s) - np.min(s),
            np.sqrt(np.mean(s ** 2))   # RMS
        ]
        # Features fréquentielles
        fv = np.abs(np.fft.rfft(s))
        fr = np.fft.rfftfreq(len(s), d=1/fs)
        features.append(np.sum(fv))    # énergie totale

        # Freeze Index — rapport bande freeze / bande locomotion
        # Valide car le passe-bande préserve ces deux bandes
        fz = fv[(fr >= 3)  & (fr <= 8)]
        lc = fv[(fr >= 0.5) & (fr <= 3)]
        fi = np.sum(fz**2) / (np.sum(lc**2) + 1e-9)
        features.append(fi)

    return np.array(features)

def predict_win(window: np.ndarray, model, scaler):
    w = window.copy()
    # Appliquer le passe-bande AVANT extraction des features
    for i in range(w.shape[1]):
        w[:, i] = bandpass_filter(w[:, i])
    feat   = extract_features(w).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred   = model.predict(feat_s)[0]
    proba  = model.predict_proba(feat_s)[0][1]
    return int(pred), round(float(proba) * 100, 1)

# ============================================
# SCORE DE RISQUE
# ============================================
def calc_score(cadence, variab, fi, prob_fog, n_fog, duree):
    s = 0
    if cadence < 90:   s += 20
    elif cadence < 100: s += 10
    if variab > 0.5:   s += 25
    elif variab > 0.2: s += 12
    if fi > 3.0:       s += 20
    elif fi > 1.5:     s += 10
    s += int(prob_fog * 0.20)
    if duree > 0:
        freq = (n_fog / duree) * 60
        if freq > 0.5:  s += 15
        elif freq > 0.2: s += 8
    return min(int(s), 100)

def risk_level(score):
    if score < 30:  return "Faible", "low",  "#27AE60"
    if score < 65:  return "Modéré", "med",  "#E67E22"
    return "Élevé",  "high", "#C0392B"

# ============================================
# INTERPRÉTATION CLINIQUE
# ============================================
def interpret_signal(cadence, variab, fi, fog_pred,
                     proba_moy, score, labels_real):
    n_fog_real = sum(labels_real)
    fog_pct    = (n_fog_real / max(len(labels_real), 1)) * 100
    anomalies  = []

    if cadence < 90:
        anomalies.append(
            f"La cadence est réduite à <b>{cadence:.0f} pas/min</b> "
            "(valeur de référence ≈ 110 ppm). Sur la courbe, "
            "cela se traduit par un espacement élargi et irrégulier "
            "entre les impacts successifs du talon au sol, visibles "
            "comme des pics d'accélération plus espacés."
        )
    if variab > 0.2:
        anomalies.append(
            f"La variabilité inter-foulée est élevée "
            f"({variab:.3f} s ; seuil pathologique &gt; 0.05 s). "
            "Sur la courbe temporelle, les cycles de marche présentent "
            "des longueurs inégales — certains nettement plus courts "
            "ou plus longs que la moyenne. Cette irrégularité est un "
            "marqueur précoce d'instabilité locomotrice."
        )
    if fi > 1.5:
        anomalies.append(
            f"Le Freeze Index atteint <b>{fi:.2f}</b> (seuil ≥ 1.5 = "
            "pathologique). Ce biomarqueur exprime le rapport entre "
            "l'énergie spectrale dans la bande de freeze (3–8 Hz) et "
            "la bande de locomotion (0.5–3 Hz). Dans les zones rouges "
            "de la courbe, l'amplitude chute quasi à zéro : le patient "
            "est en état de gel de la marche — ses pieds ne quittent "
            "plus le sol malgré la volonté de marcher."
        )
    if fog_pct > 10:
        anomalies.append(
            f"Les <b>zones surlignées en rouge</b> sur la courbe "
            f"couvrent {fog_pct:.1f}% de la session "
            f"({n_fog_real} fenêtres de 2 s chacune). "
            "Chaque zone correspond à un épisode de Freezing of Gait "
            "annotés par des cliniciens dans le jeu de données."
        )

    if score < 30:
        conclusion = (
            "Le profil cinématique de marche est dans les limites "
            "normales attendues pour un patient parkinsonien stable. "
            "Les paramètres temporels, fréquentiels et la variabilité "
            "sont cohérents avec une démarche bien compensée. "
            "Un suivi périodique de routine est suffisant."
        )
    elif score < 65:
        conclusion = (
            "Plusieurs indicateurs convergent vers une instabilité "
            "locomotrice modérée. La réduction de cadence et "
            "l'élévation de la variabilité suggèrent une dégradation "
            "progressive du contrôle moteur. Un ajustement du "
            "protocole thérapeutique et une consultation neurologique "
            "sont conseillés."
        )
    else:
        conclusion = (
            "Le profil de marche est cliniquement préoccupant. "
            "La présence simultanée d'un Freeze Index élevé, "
            "d'une forte variabilité et d'épisodes FoG fréquents "
            "indique un risque de chute immédiat. Une consultation "
            "neurologique urgente est fortement recommandée, "
            "avec révision du traitement dopaminergique."
        )
    return anomalies, conclusion

# ============================================
# RAPPORT PDF
# ============================================
def gen_pdf(patient, age, sexe, date, duree, n_fog, n_normal,
            score, cadence, variab, fi, user):
    pdf = FPDF(); pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    niveau, _, _ = risk_level(score)

    # En-tête
    pdf.set_fill_color(13, 33, 55); pdf.rect(0, 0, 210, 36, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(15, 10); pdf.cell(0, 8, "NeuroGait — Rapport d'Analyse", ln=True)
    pdf.set_font("Helvetica", "", 9); pdf.set_xy(15, 21)
    pdf.cell(0, 6, "Détection du Freezing of Gait — PFE Ingénierie Biomédicale 2026")

    pdf.set_xy(15, 44)
    pdf.set_text_color(13, 33, 55); pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Dossier Patient", ln=True)
    pdf.set_draw_color(46, 134, 193); pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(3)
    for lbl, val in [("Patient", patient),("Âge", f"{age} ans"),
                     ("Sexe", sexe),("Date", date),("Utilisateur", user)]:
        pdf.set_font("Helvetica","B",9); pdf.cell(55,5.5,f"{lbl} :",ln=False)
        pdf.set_font("Helvetica","",9); pdf.cell(0,5.5,val,ln=True)

    pdf.ln(4); pdf.set_font("Helvetica","B",11)
    pdf.cell(0,7,"Score de Risque",ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)
    pdf.set_fill_color(225,232,240); pdf.rect(15,pdf.get_y(),180,9,'F')
    if score<30:   pdf.set_fill_color(26,86,50)
    elif score<65: pdf.set_fill_color(120,66,18)
    else:          pdf.set_fill_color(100,30,22)
    pdf.rect(15,pdf.get_y(),int(180*score/100),9,'F')
    pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",9)
    pdf.set_xy(15,pdf.get_y()+1)
    pdf.cell(180,7,f"  {score}/100 — Risque {niveau}",ln=True)
    pdf.ln(2); pdf.set_text_color(13,33,55); pdf.set_font("Helvetica","",9)
    if score<30:
        pdf.multi_cell(0,5,"Profil de marche dans les limites normales. Surveillance de routine.")
    elif score<65:
        pdf.multi_cell(0,5,"Instabilité modérée détectée. Consultation neurologique conseillée.")
    else:
        pdf.multi_cell(0,5,"RISQUE ÉLEVÉ. Consultation neurologique urgente recommandée.")

    pdf.ln(4); pdf.set_font("Helvetica","B",11)
    pdf.cell(0,7,"Biomarqueurs",ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)
    hdrs=["Paramètre","Valeur","Référence","Statut"]; cw=[68,35,47,35]
    pdf.set_fill_color(13,33,55); pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",9)
    for i,h in enumerate(hdrs): pdf.cell(cw[i],7,h,border=1,fill=True)
    pdf.ln()
    rows=[("Durée session",f"{duree:.0f} s","—","—"),
          ("Cadence",f"{cadence:.1f} ppm","~110 ppm",
           "Anormal" if cadence<90 else "Normal"),
          ("Variabilité",f"{variab:.4f} s","< 0.05 s",
           "Anormal" if variab>0.2 else "Normal"),
          ("Freeze Index",f"{fi:.3f}","< 1.5",
           "Anormal" if fi>1.5 else "Normal"),
          ("Épisodes FoG",str(n_fog),"0",
           "Présent" if n_fog>0 else "Absent"),
          ("Fenêtres normales",str(n_normal),"—","—")]
    pdf.set_font("Helvetica","",9)
    for i,row in enumerate(rows):
        if i%2==0: pdf.set_fill_color(247,249,252)
        else: pdf.set_fill_color(255,255,255)
        pdf.set_text_color(13,33,55)
        for j,c in enumerate(row):
            if j==3 and c in ["Anormal","Présent"]:
                pdf.set_text_color(146,43,33)
            elif j==3: pdf.set_text_color(26,86,50)
            pdf.cell(cw[j],6,c,border=1,fill=True)
            pdf.set_text_color(13,33,55)
        pdf.ln()

    pdf.set_y(-27); pdf.set_draw_color(46,134,193)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)
    pdf.set_font("Helvetica","",8); pdf.set_text_color(150,160,170)
    pdf.cell(0,5,f"Généré le {date} — NeuroGait · PFE Ingénierie Biomédicale 2026",
             ln=True, align='C')
    pdf.cell(0,5,"Ce document est généré automatiquement. Il ne remplace pas un diagnostic médical.",
             ln=True, align='C')
    return bytes(pdf.output())

# ============================================
# GMAIL
# ============================================
def send_gmail(dest, patient, n_fog, score, gu, gp, pdf_b=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = gu; msg['To'] = dest
        niveau, _, _ = risk_level(score)
        msg['Subject'] = f"Alerte NeuroGait — {patient} · Score {score}/100"
        body = (f"Bonjour,\n\nLe système NeuroGait signale une activité "
                f"de marche nécessitant votre attention.\n\n"
                f"Patient      : {patient}\n"
                f"Date         : {datetime.now().strftime('%d/%m/%Y à %H:%M')}\n"
                f"Épisodes FoG : {n_fog}\n"
                f"Score        : {score}/100 — Risque {niveau}\n\n"
                f"Le rapport PDF est joint à ce message.\n\n"
                f"Cordialement,\nSystème NeuroGait — PFE Ingénierie Biomédicale")
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        if pdf_b:
            p = MIMEBase('application', 'octet-stream')
            p.set_payload(pdf_b); encoders.encode_base64(p)
            p.add_header('Content-Disposition',
                         f'attachment; filename="rapport_{patient}.pdf"')
            msg.attach(p)
        srv = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        srv.login(gu, gp); srv.sendmail(gu, dest, msg.as_string())
        srv.quit()
        return True, "Email envoyé avec succès."
    except Exception as e:
        return False, str(e)

# ============================================
# PAGE LOGIN
# ============================================
if not st.session_state['authenticated']:
    st.markdown("""<style>
    [data-testid="stSidebar"]{display:none!important;}
    .block-container{padding:0!important;max-width:100%!important;}
    </style>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-bottom:6px;font-size:2.2rem;'>🔬</div>
        <div style='font-family:Libre Baskerville,serif;font-size:1.85rem;
                    font-weight:700;color:#0D2137;text-align:center;margin-bottom:3px;'>
            NeuroGait
        </div>
        <div style='font-size:.82rem;color:#5D7A8A;text-align:center;
                    margin-bottom:28px;line-height:1.55;'>
            Système d'Analyse de la Marche Parkinsonienne<br>
            PFE — Ingénierie Biomédicale
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div style='background:white;border-radius:10px;padding:30px 28px 24px;
                        box-shadow:0 4px 24px rgba(13,33,55,0.11);
                        border:1px solid #D6E4F0;'>
            </div>""", unsafe_allow_html=True)
            username = st.text_input("Identifiant",
                                     placeholder="Entrez votre identifiant")
            password = st.text_input("Mot de passe", type="password",
                                     placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Connexion", use_container_width=True):
                if check_password(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")

        st.markdown("""
        <div style='text-align:center;margin-top:18px;font-size:.73rem;
                    color:#B0BEC5;line-height:1.7;'>
            Accès réservé au personnel médical autorisé<br>© 2026 NeuroGait
        </div>""", unsafe_allow_html=True)

        with st.expander("Comptes de démonstration"):
            st.markdown("""
| Identifiant | Mot de passe |
|---|---|
| `admin` | `admin123` |
| `medecin` | `parkinson2026` |
| `lina` | `pfe2026` |""")
    st.stop()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown(f"""
    <div style='padding:22px 0 14px;text-align:center;'>
        <div style='font-size:2rem;'>🔬</div>
        <div style='font-family:Libre Baskerville,serif;font-size:1.1rem;
                    font-weight:700;color:white;margin-top:5px;'>NeuroGait</div>
        <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                    letter-spacing:2px;text-transform:uppercase;'>
            Analyse de Marche
        </div>
    </div><hr>
    <div style='background:rgba(255,255,255,.07);border-radius:6px;
                padding:8px 12px;margin-bottom:14px;'>
        <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                    text-transform:uppercase;letter-spacing:1px;'>Session</div>
        <div style='font-size:.88rem;font-weight:600;color:white;margin-top:2px;'>
            {st.session_state['username']}
        </div>
    </div>
    <div class='sec-lbl' style='color:rgba(255,255,255,.32);
         border-color:rgba(255,255,255,.07);'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Analyse Fichier", "Démo Temps Réel",
                         "Historique", "Configuration"],
                    label_visibility="collapsed")

    st.markdown("""<hr><div class='sec-lbl'
    style='color:rgba(255,255,255,.32);border-color:rgba(255,255,255,.07);'>
    Dossier Patient</div>""", unsafe_allow_html=True)

    nom  = st.text_input("Nom", "Patient 001", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1: age  = st.number_input("Âge",  0, 120, 65, label_visibility="collapsed")
    with c2: sexe = st.selectbox("Sexe", ["H","F"], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    if model_ok:
        st.markdown("<div style='background:rgba(26,86,50,.22);border:1px solid "
                    "rgba(39,174,96,.28);border-radius:5px;padding:7px 11px;"
                    "font-size:.76rem;color:#7DCEA0;font-weight:600;'>"
                    "Modèle ML chargé</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:rgba(146,43,33,.2);border:1px solid "
                    "rgba(192,57,43,.28);border-radius:5px;padding:7px 11px;"
                    "font-size:.76rem;color:#F1948A;font-weight:600;'>"
                    "Modèle non trouvé</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Déconnexion", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ''
        st.rerun()

# ============================================
# HEADER
# ============================================
st.markdown(f"""
<div class="page-header">
    <div class="ph-user">{st.session_state['username']}</div>
    <div class="ph-eyebrow">Système de Diagnostic Médical</div>
    <div class="ph-title">Analyse de la Marche Parkinsonienne</div>
    <div class="ph-sub">
        Détection FoG · Score de Risque · Interprétation Clinique · Rapport PDF
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE — ANALYSE FICHIER
# ============================================
if page == "Analyse Fichier":

    st.markdown('<div class="sec-lbl">Chargement des données</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Fichier .txt (Daphnet) ou .csv",
                                type=["txt","csv"],
                                label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div style='background:white;border-radius:10px;padding:52px;
                    text-align:center;border:1px solid #D6E4F0;'>
            <div style='font-size:2.6rem;margin-bottom:12px;'>📂</div>
            <div style='font-family:Libre Baskerville,serif;font-size:1.2rem;
                        font-weight:700;color:#0D2137;margin-bottom:7px;'>
                Aucun fichier chargé
            </div>
            <div style='color:#5D7A8A;font-size:.86rem;max-width:320px;
                        margin:0 auto;line-height:1.6;'>
                Glissez un fichier <b>.txt</b> du dataset Daphnet FoG
                pour démarrer l'analyse.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    col_names = ['time','ankle_x','ankle_y','ankle_z',
                 'thigh_x','thigh_y','thigh_z',
                 'trunk_x','trunk_y','trunk_z','label']
    try:
        df = pd.read_csv(uploaded, sep=" ", header=None, names=col_names)
        df = df[df['label'] != 0].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erreur de lecture : {e}"); st.stop()

    # Pipeline
    WS, OL = 128, 64
    windows_t, labels_pred, labels_real, probas = [], [], [], []

    for start in range(0, len(df) - WS, OL):
        end = start + WS
        win = df[['ankle_x','ankle_y','ankle_z']].iloc[start:end].values
        lbl = df['label'].iloc[start:end].mode()[0]
        if model_ok:
            pred, prob = predict_win(win, model, scaler)
            labels_pred.append(pred); probas.append(prob)
        labels_real.append(1 if lbl == 2 else 0)
        windows_t.append(start / 64)

    fog_pred  = sum(labels_pred) if model_ok else 0
    fog_real  = sum(labels_real)
    duree     = len(df) / 64
    az        = df['ankle_z'].values

    # Biomarqueurs sur signal passe-bande
    az_bp = bandpass_filter(az)
    try:
        peaks, _ = find_peaks(az_bp, distance=32)
        if len(peaks) > 2:
            diffs   = np.diff(peaks) / 64
            cadence = float(60 / np.mean(diffs))
            variab  = float(np.std(diffs))
        else:
            cadence, variab = 100.0, 0.1
    except Exception:
        cadence, variab = 100.0, 0.1
    cadence = min(max(cadence, 50), 150)
    variab  = min(variab, 2.0)

    N     = min(512, len(az_bp))
    fv    = np.abs(np.fft.rfft(az_bp[:N]))
    fr    = np.fft.rfftfreq(N, d=1/64)
    fz    = fv[(fr >= 3) & (fr <= 8)]
    lc_   = fv[(fr >= 0.5) & (fr <= 3)]
    fi    = float(np.sum(fz**2) / (np.sum(lc_**2) + 1e-9))
    fi    = min(fi, 10.0)

    p_avg = float(np.mean(probas)) if probas else 0.0
    score = calc_score(cadence, variab, fi, p_avg/100, fog_pred, duree)
    niv, sc_cls, sc_col = risk_level(score)
    anomalies, conclusion = interpret_signal(cadence, variab, fi,
                                             fog_pred, p_avg, score,
                                             labels_real)

    # KPI
    cols = st.columns(5)
    kpis = [
        ("", "Durée session", f"{duree:.0f} s", f"{len(df):,} points"),
        ("ok" if fog_pred==0 else "dk",
         "Marche normale",
         str(len(labels_pred)-fog_pred if labels_pred else 0),
         "fenêtres saines"),
        ("dk", "Épisodes FoG", str(fog_pred), "détectés par IA"),
        ("wk" if cadence<100 else "ok",
         "Cadence", f"{cadence:.0f}", "pas / min"),
        ("dk" if score>=65 else ("wk" if score>=30 else "ok"),
         "Score de risque", str(score), f"Risque {niv}"),
    ]
    for col, (cls, lbl, val, note) in zip(cols, kpis):
        with col:
            st.markdown(f'<div class="kpi {cls}"><div class="kpi-lbl">{lbl}</div>'
                        f'<div class="kpi-val">{val}</div>'
                        f'<div class="kpi-note">{note}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_main, col_side = st.columns([3, 1])

    with col_main:
        # Signal annoté
        st.markdown('<div class="sec-lbl">Signal accéléromètre — cheville (axe Z)</div>',
                    unsafe_allow_html=True)

        az_viz = lowpass_viz(az)   # version lissée pour affichage uniquement
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=az_viz[:3000], mode='lines',
            name='Signal (lissé — visualisation)',
            line=dict(color='#1A5276', width=1.3),
            hovertemplate='%{y:.1f} mg<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            y=az[:3000], mode='lines', name='Signal brut',
            line=dict(color='#AED6F1', width=0.6, dash='dot'),
            opacity=0.4,
            hovertemplate='Brut : %{y:.1f} mg<extra></extra>'
        ))

        in_fog = False; sf = 0; n_annot = 0
        for i in range(min(3000, len(df))):
            if df['label'].iloc[i] == 2 and not in_fog:
                sf = i; in_fog = True
            elif df['label'].iloc[i] != 2 and in_fog:
                fig.add_vrect(x0=sf, x1=i, fillcolor="#C0392B",
                              opacity=0.10, line_width=0)
                if n_annot == 0:
                    fig.add_annotation(
                        x=(sf+i)//2, y=float(np.max(az_viz[:3000]))*0.95,
                        text="Épisode FoG<br>(gel de la marche)",
                        showarrow=True, arrowhead=2,
                        arrowcolor="#922B21", arrowsize=1.1,
                        font=dict(size=9, color="#922B21"),
                        ax=0, ay=-32
                    )
                n_annot += 1; in_fog = False

        if len(az_viz[:3000]) > 60:
            fig.add_annotation(
                x=25, y=float(np.percentile(az_viz[:3000], 20)),
                text="Marche normale :<br>pics réguliers",
                showarrow=False,
                font=dict(size=9, color="#1A5276"),
                bgcolor="rgba(235,245,251,.88)",
                bordercolor="#AED6F1", borderwidth=1, borderpad=4
            )

        fig.update_layout(
            height=295,
            margin=dict(l=5, r=5, t=5, b=5),
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(color='#1A252F', family='Source Sans 3', size=11),
            xaxis=dict(title="Échantillons", showgrid=True,
                       gridcolor='#F0F4F8', linecolor='#D6E4F0',
                       tickfont=dict(size=10), zeroline=False),
            yaxis=dict(title="mg", showgrid=True,
                       gridcolor='#F0F4F8', linecolor='#D6E4F0',
                       tickfont=dict(size=10), zeroline=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.01,
                        font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                        borderwidth=0),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False,
                                'responsive': True})

        # Courbe de probabilité
        if model_ok and probas:
            st.markdown('<div class="sec-lbl">Probabilité FoG au fil du temps</div>',
                        unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=windows_t, y=probas, mode='lines',
                fill='tozeroy',
                line=dict(color='#1A5276', width=1.4),
                fillcolor='rgba(26,82,118,.07)',
                hovertemplate='t=%{x:.0f}s  P(FoG)=%{y:.1f}%<extra></extra>'
            ))
            fig2.add_hrect(y0=60, y1=105,
                           fillcolor="rgba(192,57,43,.05)", line_width=0)
            fig2.add_hline(y=60, line_dash="dash",
                           line_color="#C0392B", line_width=1,
                           annotation_text="Seuil 60 %",
                           annotation_font=dict(size=9, color="#922B21"))
            fig2.update_layout(
                height=175,
                margin=dict(l=5, r=5, t=5, b=5),
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(color='#1A252F', family='Source Sans 3', size=11),
                xaxis=dict(title="Temps (s)", showgrid=True,
                           gridcolor='#F0F4F8', zeroline=False,
                           tickfont=dict(size=10)),
                yaxis=dict(title="%", range=[0, 105], showgrid=True,
                           gridcolor='#F0F4F8', zeroline=False,
                           tickfont=dict(size=10)),
                showlegend=False, hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True,
                            config={'displayModeBar': False,
                                    'responsive': True})

    with col_side:
        # Score
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-title">Score de risque</div>
            <div class="score-ring {sc_cls}">
                <div class="score-num">{score}</div>
                <div class="score-denom">/100</div>
            </div>
            <div class="score-title">Risque {niv}</div>
            <div class="score-sub">{datetime.now().strftime("%d/%m/%Y")}</div>
        </div>""", unsafe_allow_html=True)

        # Features
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Biomarqueurs</div>',
                    unsafe_allow_html=True)
        tag_l = {"ok": "Normal", "med": "Modéré", "nok": "Anormal"}
        feat_rows = [
            ("Cadence", f"{cadence:.1f} ppm", "~110 ppm",
             "ok" if cadence>=100 else ("med" if cadence>=85 else "nok")),
            ("Variabilité", f"{variab:.4f} s", "< 0.05 s",
             "ok" if variab<0.05 else ("med" if variab<0.2 else "nok")),
            ("Freeze Index", f"{fi:.3f}", "< 1.5",
             "ok" if fi<1.5 else ("med" if fi<3.0 else "nok")),
            ("P(FoG) moy.", f"{p_avg:.1f}%", "< 30%",
             "ok" if p_avg<30 else ("med" if p_avg<60 else "nok")),
            ("FoG détectés", str(fog_pred), "0",
             "ok" if fog_pred==0 else ("med" if fog_pred<=3 else "nok")),
        ]
        for fn, fv_, fr_, ft in feat_rows:
            st.markdown(f"""
            <div class="feat-row">
                <div>
                    <div class="feat-name">{fn}</div>
                    <div class="feat-ref">Réf : {fr_}</div>
                </div>
                <div style="text-align:right;">
                    <div class="feat-val">{fv_}</div>
                    <span class="tag {ft}">{tag_l[ft]}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interprétation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Interprétation clinique du signal</div>',
                unsafe_allow_html=True)
    if anomalies:
        for i, a in enumerate(anomalies):
            crit = "FoG" in a or "Freeze" in a or "rouge" in a
            css = "crit" if crit else ""
            st.markdown(f"""
            <div class="interp-box {css}">
                <div class="interp-title {css}">Observation {i+1}</div>
                {a}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="interp-box">
            <div class="interp-title">Observations</div>
            Le signal présente un profil rythmique régulier. Les pics
            d'accélération sont espacés uniformément, traduisant un
            cycle de marche stable, sans anomalie notable.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box {'crit' if score>=65 else ''}">
        <div class="interp-title {'crit' if score>=65 else ''}">
            Conclusion générale
        </div>
        {conclusion}
    </div>""", unsafe_allow_html=True)

    # Actions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Actions</div>', unsafe_allow_html=True)
    pdf_b = gen_pdf(nom, age, sexe,
                    datetime.now().strftime("%d/%m/%Y %H:%M"),
                    duree, fog_pred,
                    len(labels_pred)-fog_pred if labels_pred else 0,
                    score, cadence, variab, fi,
                    st.session_state['username'])
    a1, a2 = st.columns(2)
    with a1:
        st.download_button(
            "Télécharger le rapport PDF", data=pdf_b,
            file_name=f"rapport_{nom}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf", use_container_width=True)
    with a2:
        if st.button("Sauvegarder dans l'historique",
                     use_container_width=True):
            save_session(nom, age, sexe, duree, fog_pred,
                         len(labels_pred)-fog_pred if labels_pred else 0,
                         score, cadence, variab, fi,
                         st.session_state['username'])
            st.success("Session enregistrée.")
    if score >= 65 and fog_pred > 0:
        st.markdown(f"""
        <div class="alert-crit">
            <b>Risque élevé détecté</b>
            <small>{fog_pred} épisodes FoG · Score {score}/100 ·
            Configurez les alertes email dans l'onglet Configuration.</small>
        </div>""", unsafe_allow_html=True)

# ============================================
# PAGE — DÉMO TEMPS RÉEL
# CORRECTION 2 : streaming de données Daphnet
# réelles au lieu de bruit aléatoire
# ============================================
elif page == "Démo Temps Réel":

    st.markdown("""
    <div class="interp-box" style="margin-bottom:16px;">
        <div class="interp-title">Mode de simulation</div>
        Cette démonstration injecte les données d'un patient réel
        du dataset Daphnet FoG ligne par ligne, reproduisant
        fidèlement les conditions d'acquisition en temps réel.
        Chargez un fichier pour démarrer.
    </div>""", unsafe_allow_html=True)

    demo_file = st.file_uploader(
        "Fichier patient pour la simulation (format Daphnet .txt)",
        type=["txt"], key="demo_upload",
        label_visibility="collapsed"
    )

    if demo_file is not None:
        col_names = ['time','ankle_x','ankle_y','ankle_z',
                     'thigh_x','thigh_y','thigh_z',
                     'trunk_x','trunk_y','trunk_z','label']
        df_demo = pd.read_csv(demo_file, sep=" ",
                              header=None, names=col_names)
        df_demo = df_demo[df_demo['label'] != 0].reset_index(drop=True)
        st.session_state['demo_df'] = df_demo
        st.success(f"Fichier chargé : {len(df_demo)} échantillons · "
                   f"{(df_demo['label']==2).sum()} points FoG annotés")

    col_c, col_d = st.columns([3, 1])
    with col_d:
        st.markdown('<div class="sec-lbl">Diagnostic</div>',
                    unsafe_allow_html=True)
        dx_box  = st.empty()
        pb_box  = st.empty()
        st_box  = st.empty()
        dx_box.markdown('<div class="dx wait"><div class="dx-icon">◌</div>'
                        '<div class="dx-title">En attente</div>'
                        '<div class="dx-sub">Démarrez la session</div></div>',
                        unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="sec-lbl">Signal IMU — temps réel</div>',
                    unsafe_allow_html=True)
        ch_box = st.empty()

    c1, c2, _ = st.columns([1, 1, 5])
    with c1:
        s1 = st.button("Démarrer", type="primary",
                       use_container_width=True)
    with c2:
        s2 = st.button("Arrêter", use_container_width=True)
    if s1: st.session_state['running'] = True
    if s2: st.session_state['running'] = False

    if st.session_state.get('running', False):
        df_src = st.session_state.get('demo_df', None)
        if df_src is None:
            st.warning("Chargez d'abord un fichier patient ci-dessus.")
            st.session_state['running'] = False
        else:
            buf = collections.deque(maxlen=128)
            hx, hy, hz, hlbl = [], [], [], []
            nf = nn = 0; lp = 0; lpr = 0.0
            idx = st.session_state.get('demo_idx', 0)

            while st.session_state.get('running', False):
                if idx >= len(df_src):
                    idx = 0   # boucle sur le fichier

                row = df_src.iloc[idx]
                ax_, ay_, az_ = (float(row['ankle_x']),
                                 float(row['ankle_y']),
                                 float(row['ankle_z']))
                real_lbl = int(row['label'])

                buf.append([ax_, ay_, az_])
                hx.append(ax_); hy.append(ay_)
                hz.append(az_); hlbl.append(real_lbl)

                N_SHOW = 320
                if len(hx) > N_SHOW:
                    hx=hx[-N_SHOW:]; hy=hy[-N_SHOW:]
                    hz=hz[-N_SHOW:]; hlbl=hlbl[-N_SHOW:]

                if len(buf) == 128 and model_ok:
                    lp, lpr = predict_win(np.array(buf), model, scaler)
                    if lp == 1: nf += 1
                    else: nn += 1

                # Diagnostic
                if lp == 1:
                    dx_box.markdown(
                        '<div class="dx fog"><div class="dx-icon">●</div>'
                        '<div class="dx-title">FoG détecté</div>'
                        '<div class="dx-sub">Enrayement cinétique</div>'
                        '</div>', unsafe_allow_html=True)
                else:
                    dx_box.markdown(
                        '<div class="dx normal"><div class="dx-icon">●</div>'
                        '<div class="dx-title">Marche normale</div>'
                        '<div class="dx-sub">Aucune anomalie</div>'
                        '</div>', unsafe_allow_html=True)

                bc = "#C0392B" if lpr > 60 else "#1A5276"
                pb_box.markdown(f"""
                <div class="pb-wrap">
                    <div class="pb-lbl">Probabilité FoG</div>
                    <div class="pb-bg">
                        <div class="pb-fill"
                             style="width:{lpr}%;background:{bc};"></div>
                    </div>
                    <div class="pb-num" style="color:{bc};">{lpr}%</div>
                </div>""", unsafe_allow_html=True)

                st_box.markdown(f"""
                <div class="card">
                    <div class="feat-row">
                        <span class="feat-name">Normal</span>
                        <span class="feat-val" style="color:#1A5632;">{nn}</span>
                    </div>
                    <div class="feat-row">
                        <span class="feat-name">FoG</span>
                        <span class="feat-val" style="color:#922B21;">{nf}</span>
                    </div>
                    <div class="feat-row" style="border:none;">
                        <span class="feat-name" style="font-size:.75rem;
                              color:var(--muted);">Label réel</span>
                        <span class="tag {'nok' if real_lbl==2 else 'ok'}">
                            {'FoG' if real_lbl==2 else 'Normal'}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Graphique
                is_fog_now = (real_lbl == 2)
                lc_ = '#C0392B' if is_fog_now else '#1A5276'
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hx, mode='lines', name='X',
                    line=dict(color='#AED6F1', width=0.9)))
                fig.add_trace(go.Scatter(y=hy, mode='lines', name='Y',
                    line=dict(color='#A9DFBF', width=0.9)))
                fig.add_trace(go.Scatter(y=hz, mode='lines', name='Z',
                    line=dict(color=lc_, width=1.7)))
                fig.update_layout(
                    height=305,
                    margin=dict(l=5, r=5, t=5, b=5),
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(color='#1A252F',
                              family='Source Sans 3', size=11),
                    xaxis=dict(showgrid=True, gridcolor='#F0F4F8',
                               zeroline=False, tickfont=dict(size=10)),
                    yaxis=dict(title='mg', showgrid=True,
                               gridcolor='#F0F4F8', zeroline=False,
                               tickfont=dict(size=10)),
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.01, font=dict(size=10),
                                bgcolor='rgba(0,0,0,0)', borderwidth=0)
                )
                ch_box.plotly_chart(fig, use_container_width=True,
                                    config={'displayModeBar': False,
                                            'responsive': True})
                idx += 1
                st.session_state['demo_idx'] = idx
                time.sleep(1/64)   # respect de la fréquence 64 Hz

    elif not st.session_state.get('demo_df') is None:
        with col_c:
            st.markdown("""
            <div style='background:white;border-radius:9px;padding:44px;
                        text-align:center;border:1px solid #D6E4F0;'>
                <div style='font-size:2.4rem;margin-bottom:10px;'>▶</div>
                <div style='font-family:Libre Baskerville,serif;font-size:1.1rem;
                            font-weight:700;color:#0D2137;'>
                    Cliquez sur Démarrer
                </div>
            </div>""", unsafe_allow_html=True)

# ============================================
# PAGE — HISTORIQUE
# ============================================
elif page == "Historique":

    st.markdown('<div class="sec-lbl">Historique des sessions</div>',
                unsafe_allow_html=True)
    try:
        df_h = get_sessions()
        if df_h.empty:
            st.info("Aucune session. Analysez un fichier et "
                    "cliquez sur Sauvegarder.")
        else:
            cols = st.columns(4)
            for col, cls, lbl, val, note in [
                (cols[0], "",   "Sessions",
                 len(df_h), "enregistrées"),
                (cols[1], "dk", "FoG total",
                 int(df_h['fog_episodes'].sum()), "épisodes"),
                (cols[2], "wk", "Score moyen",
                 f"{df_h['score_risque'].mean():.0f}", "/100"),
                (cols[3], "ok", "Patients",
                 df_h['patient'].nunique(), "uniques"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="kpi {cls}"><div class="kpi-lbl">'
                        f'{lbl}</div><div class="kpi-val">{val}</div>'
                        f'<div class="kpi-note">{note}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-lbl">Évolution du score de risque</div>',
                        unsafe_allow_html=True)

            fig_h = go.Figure()
            for pt in df_h['patient'].unique():
                dp = df_h[df_h['patient'] == pt]
                fig_h.add_trace(go.Scatter(
                    x=dp['date'], y=dp['score_risque'],
                    mode='lines+markers', name=pt,
                    line=dict(width=2)))
            fig_h.add_hline(y=65, line_dash="dash",
                            line_color="#C0392B", line_width=1,
                            annotation_text="Seuil risque élevé",
                            annotation_font=dict(size=9, color="#922B21"))
            fig_h.update_layout(
                height=250,
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(color='#1A252F', family='Source Sans 3', size=11),
                xaxis=dict(title="Date", showgrid=True,
                           gridcolor='#F0F4F8', zeroline=False),
                yaxis=dict(title="Score", range=[0, 105],
                           showgrid=True, gridcolor='#F0F4F8',
                           zeroline=False),
                margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig_h, use_container_width=True,
                            config={'displayModeBar': False})

            st.markdown('<div class="sec-lbl">Détail</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                df_h[['patient','age','date','duree',
                      'fog_episodes','score_risque','utilisateur']]
                .rename(columns={
                    'patient':'Patient','age':'Âge','date':'Date',
                    'duree':'Durée (s)','fog_episodes':'FoG',
                    'score_risque':'Score','utilisateur':'Utilisateur'}),
                use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

# ============================================
# PAGE — CONFIGURATION
# ============================================
elif page == "Configuration":

    st.markdown('<div class="sec-lbl">Alertes email (Gmail)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="interp-box">
        <div class="interp-title">Configuration requise</div>
        Activez la validation en deux étapes sur votre compte Gmail,
        puis générez un <b>mot de passe d'application</b> dans
        Paramètres → Sécurité → Mots de passe des applications.
        Utilisez ce code de 16 caractères ci-dessous — et non votre
        mot de passe habituel.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expéditeur**")
        gu = st.text_input("Adresse Gmail",
                           placeholder="votre.email@gmail.com")
        gp = st.text_input("Mot de passe d'application",
                           type="password",
                           placeholder="xxxx xxxx xxxx xxxx")
    with c2:
        st.markdown("**Destinataire**")
        de = st.text_input("Email médecin",
                           placeholder="medecin@hopital.ma")
        st.slider("Alerte si FoG ≥", 1, 10, 3)
        st.slider("Alerte si Score ≥", 30, 100, 65)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Envoyer un email de test"):
        if not gu or not gp or not de:
            st.error("Remplissez tous les champs.")
        else:
            pb_ = gen_pdf("Patient Test", 65, "H",
                          datetime.now().strftime("%d/%m/%Y %H:%M"),
                          120, 5, 45, 72, 88.5, .45, 2.8,
                          st.session_state['username'])
            ok_, msg_ = send_gmail(de, "Patient Test", 5, 72, gu, gp, pb_)
            if ok_: st.success(msg_)
            else: st.error(msg_)

    st.markdown("---")
    st.markdown("**Envoi manuel d'une alerte**")
    m1, m2 = st.columns(2)
    with m1:
        pa = st.text_input("Patient", nom)
        fa = st.number_input("Épisodes FoG", 0, 100, 5)
    with m2:
        sa = st.number_input("Score de risque", 0, 100, 72)

    if st.button("Envoyer l'alerte", type="primary"):
        if not gu or not gp or not de:
            st.error("Configurez d'abord Gmail ci-dessus.")
        else:
            pb_ = gen_pdf(pa, age, sexe,
                          datetime.now().strftime("%d/%m/%Y %H:%M"),
                          0, fa, 0, sa, 100., .5, 2.,
                          st.session_state['username'])
            ok_, msg_ = send_gmail(de, pa, fa, sa, gu, gp, pb_)
            if ok_: st.success(msg_)
            else: st.error(msg_)# ============================================
# app_cloud.py — NeuroGait
# Version corrigée — Remarques jury PFE
# Corrections :
#   1. Filtre passe-bande 0.5–20 Hz (préserve bande FoG 3–8 Hz)
#   2. Simulation réaliste via données Daphnet réelles
#   3. SQLite avec gestionnaire de contexte
#   4. UI accessible — contrastes corrects
# ============================================
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
import os
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
# UTILISATEURS
# ============================================
USERS = {
    "admin":   hashlib.sha256("admin123".encode()).hexdigest(),
    "medecin": hashlib.sha256("parkinson2026".encode()).hexdigest(),
    "lina":    hashlib.sha256("pfe2026".encode()).hexdigest(),
}
def check_password(u, p):
    return USERS.get(u) == hashlib.sha256(p.encode()).hexdigest()

# ============================================
# BASE DE DONNÉES — Gestionnaire de contexte
# ============================================
DB_PATH = "neurogait.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                patient       TEXT,
                age           INTEGER,
                sexe          TEXT,
                date          TEXT,
                duree         REAL,
                fog_episodes  INTEGER,
                normal_windows INTEGER,
                score_risque  INTEGER,
                cadence       REAL,
                variabilite   REAL,
                freeze_index  REAL,
                utilisateur   TEXT
            )
        """)

def save_session(patient, age, sexe, duree, n_fog, n_normal,
                 score, cadence, variab, fi, user):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO sessions
            (patient,age,sexe,date,duree,fog_episodes,
             normal_windows,score_risque,cadence,
             variabilite,freeze_index,utilisateur)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (patient, age, sexe,
              datetime.now().strftime("%d/%m/%Y %H:%M"),
              round(duree,1), n_fog, n_normal, score,
              round(cadence,1), round(variab,4),
              round(fi,3), user))

def get_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(
            "SELECT * FROM sessions ORDER BY id DESC", conn
        )

init_db()

# ============================================
# CSS — Médical bleu/blanc, contrastes corrects
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0D2137;
    --blue:   #1A5276;
    --mid:    #2E86C1;
    --light:  #AED6F1;
    --pale:   #EBF5FB;
    --white:  #FFFFFF;
    --grey:   #F7F9FC;
    --border: #D6E4F0;
    --text:   #1A252F;
    --muted:  #5D7A8A;
    --danger: #922B21;
    --ok:     #1A5632;
    --warn:   #784212;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--grey) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    color: var(--text) !important;
}

/* Réduire padding bloc principal */
.block-container { padding-top: 1rem !important; }

/* ── SIDEBAR ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #C8D8E8 !important; }

/* Inputs sidebar */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

/* ── INPUTS PAGE PRINCIPALE ──────────────── */
/* Fond blanc, texte sombre — correction contraste jury */
.main input[type="text"],
.main input[type="password"],
.main input[type="number"],
.main textarea,
[data-testid="stTextInput"] input,
[data-testid="stPasswordInput"] input,
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stPasswordInput"] input::placeholder {
    color: #95A5A6 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stPasswordInput"] input:focus {
    border-color: var(--mid) !important;
    box-shadow: 0 0 0 3px rgba(46,134,193,0.12) !important;
}

/* Selectbox page principale */
[data-baseweb="select"] > div {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
}

/* ── PAGE HEADER ─────────────────────────── */
.page-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 65%, var(--mid) 100%);
    border-radius: 10px; padding: 28px 34px;
    margin-bottom: 22px;
    box-shadow: 0 2px 14px rgba(13,33,55,0.16);
    position: relative; overflow: hidden;
}
.page-header::after {
    content: ''; position: absolute;
    right: -28px; top: -28px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
    pointer-events: none;
}
.ph-eyebrow {
    font-size: 0.67rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--light); margin-bottom: 7px;
}
.ph-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.85rem; font-weight: 700;
    color: white; letter-spacing: -0.3px; line-height: 1.15;
}
.ph-sub {
    font-size: 0.86rem; color: rgba(255,255,255,0.58);
    margin-top: 5px; font-weight: 300;
}
.ph-user {
    position: absolute; top: 18px; right: 22px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 18px; padding: 4px 13px;
    font-size: 0.78rem; color: rgba(255,255,255,0.8);
    font-weight: 500;
}

/* ── KPI ─────────────────────────────────── */
.kpi {
    background: white; border-radius: 8px;
    padding: 15px 17px;
    box-shadow: 0 1px 5px rgba(13,33,55,0.06);
    border: 1px solid var(--border);
    border-top: 3px solid var(--mid);
}
.kpi.dk  { border-top-color: var(--danger); }
.kpi.ok  { border-top-color: var(--ok); }
.kpi.wk  { border-top-color: #E67E22; }
.kpi-lbl { font-size: 0.67rem; font-weight: 700; color: var(--muted);
           text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.kpi-val { font-family: 'Libre Baskerville', serif;
           font-size: 1.7rem; font-weight: 700; color: var(--navy); line-height: 1.1; }
.kpi-note { font-size: 0.73rem; color: var(--muted); margin-top: 2px; }

/* ── CARD ────────────────────────────────── */
.card {
    background: white; border-radius: 9px;
    padding: 18px 20px;
    box-shadow: 0 1px 7px rgba(13,33,55,0.06);
    border: 1px solid var(--border);
    margin-bottom: 14px;
}
.card-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 0.9rem; font-weight: 700;
    color: var(--navy); margin-bottom: 12px;
    padding-bottom: 9px; border-bottom: 1px solid var(--border);
}

/* ── SCORE ───────────────────────────────── */
.score-ring {
    width: 105px; height: 105px; border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin: 0 auto 10px;
    font-family: 'Libre Baskerville', serif;
    font-weight: 700; color: white;
    box-shadow: 0 3px 14px rgba(0,0,0,0.17);
}
.score-ring.low  { background: linear-gradient(135deg,#1A5632,#27AE60); }
.score-ring.med  { background: linear-gradient(135deg,#784212,#E67E22); }
.score-ring.high { background: linear-gradient(135deg,#641E16,#C0392B);
                   animation: ring-pulse 2s infinite; }
@keyframes ring-pulse {
    0%,100% { box-shadow: 0 3px 14px rgba(192,57,43,.28); }
    50%      { box-shadow: 0 3px 26px rgba(192,57,43,.62); }
}
.score-num    { font-size: 1.75rem; line-height: 1; }
.score-denom  { font-size: 0.67rem; opacity: .75; }
.score-title  { font-family: 'Libre Baskerville', serif; font-size: 0.92rem;
                font-weight: 700; color: var(--navy); text-align: center; }
.score-sub    { font-size: 0.75rem; color: var(--muted); text-align: center; }

/* ── FEATURE ROW ─────────────────────────── */
.feat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
}
.feat-row:last-child { border-bottom: none; }
.feat-name { color: var(--text); font-weight: 500; }
.feat-val  { font-weight: 700;
             font-family: 'Libre Baskerville', serif; font-size: 0.92rem; }
.feat-ref  { font-size: 0.69rem; color: var(--muted); }
.tag {
    display: inline-block; font-size: 0.65rem; font-weight: 700;
    padding: 2px 7px; border-radius: 9px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.tag.ok  { background: #EAFAF1; color: #1A5632; }
.tag.nok { background: #FDEDEC; color: #922B21; }
.tag.med { background: #FEF9E7; color: #784212; }

/* ── INTERPRÉTATION ──────────────────────── */
.interp-box {
    background: var(--pale); border: 1px solid var(--border);
    border-left: 3px solid var(--mid);
    border-radius: 0 8px 8px 0;
    padding: 13px 17px; margin-bottom: 9px;
    font-size: 0.84rem; color: var(--text); line-height: 1.65;
}
.interp-box.crit { background: #FEF5F5; border-left-color: var(--danger); }
.interp-title { font-size: 0.67rem; font-weight: 700;
                text-transform: uppercase; letter-spacing: 1px;
                color: var(--blue); margin-bottom: 4px; }
.interp-title.crit { color: var(--danger); }

/* ── DIAGNOSTIC ──────────────────────────── */
.dx {
    border-radius: 8px; padding: 16px 18px;
    text-align: center; margin-bottom: 12px;
}
.dx.fog    { background: linear-gradient(135deg,#922B21,#C0392B);
             animation: ring-pulse 2s infinite; }
.dx.normal { background: linear-gradient(135deg,#1A5632,#27AE60); }
.dx.wait   { background: linear-gradient(135deg,#1A252F,#2C3E50); }
.dx-icon   { font-size: 1.8rem; margin-bottom: 5px; }
.dx-title  { font-family: 'Libre Baskerville', serif;
             font-size: 1.1rem; font-weight: 700; color: white; }
.dx-sub    { font-size: .77rem; color: rgba(255,255,255,.7); margin-top: 2px; }

/* ── PROBA BAR ───────────────────────────── */
.pb-wrap { background: white; border-radius: 7px;
           padding: 13px 15px; border: 1px solid var(--border);
           margin-bottom: 11px; }
.pb-lbl  { font-size: .67rem; font-weight: 700; color: var(--muted);
           text-transform: uppercase; letter-spacing: 1px; margin-bottom: 7px; }
.pb-bg   { background: #E8EDF2; border-radius: 4px; height: 7px; overflow: hidden; }
.pb-fill { height: 100%; border-radius: 4px; transition: width .35s ease; }
.pb-num  { font-family: 'Libre Baskerville', serif;
           font-size: 1.35rem; font-weight: 700; margin-top: 4px; }

/* ── ALERT ───────────────────────────────── */
.alert-crit {
    background: linear-gradient(90deg,#641E16,#922B21);
    border-radius: 7px; padding: 13px 17px; color: white; margin-bottom: 12px;
}
.alert-crit b { font-weight: 700; font-size: .88rem; display: block; }
.alert-crit small { font-size: .77rem; opacity: .82; }

/* ── SECTION LABEL ───────────────────────── */
.sec-lbl {
    font-size: .66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: var(--muted); margin-bottom: 12px;
    border-bottom: 1px solid var(--border); padding-bottom: 5px;
}

/* ── BUTTONS ─────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg,var(--navy),var(--blue)) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: .86rem !important;
    padding: 8px 18px !important; transition: opacity .18s !important;
}
.stButton > button:hover { opacity: .80 !important; }
.stDownloadButton > button {
    background: white !important; color: var(--navy) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: .86rem !important;
}

/* ── HIDE STREAMLIT DEFAULT UI ───────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton,
[data-testid="stToolbar"] { display: none; }
hr { border-color: rgba(255,255,255,.08) !important; }

/* ── FILE UPLOADER ───────────────────────── */
[data-testid="stFileUploader"] {
    background: white; border-radius: 8px;
    border: 1.5px dashed var(--border) !important;
    padding: 10px;
}

/* ── DATAFRAME ───────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
for k, v in [('authenticated', False), ('username', ''),
             ('running', False), ('demo_idx', 0),
             ('demo_df', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# ML — Chargement modèle
# ============================================
@st.cache_resource
def load_model():
    m = joblib.load("models/model_fog_rf.pkl")
    s = joblib.load("models/scaler_fog.pkl")
    return m, s

try:
    model, scaler = load_model()
    model_ok = True
except Exception:
    model_ok = False

# ============================================
# CORRECTION 1 — Filtre PASSE-BANDE 0.5–20 Hz
# Préserve la bande FoG (3–8 Hz) et la bande
# locomotrice (0.5–3 Hz). Élimine :
#   – dérive basse fréquence (gravité, < 0.5 Hz)
#   – bruit électronique haute fréquence (> 20 Hz)
# ============================================
def bandpass_filter(signal: np.ndarray,
                    low: float = 0.5,
                    high: float = 20.0,
                    fs: int = 64,
                    order: int = 4) -> np.ndarray:
    """
    Filtre passe-bande Butterworth.
    Contrairement au passe-bas à 3 Hz (erreur initiale),
    ce filtre préserve intactes les deux bandes d'intérêt :
      · Locomotion : 0.5–3 Hz
      · Freeze     : 3–8 Hz  ← signature spectrale du FoG
    """
    nyq = 0.5 * fs
    low_n  = low  / nyq
    high_n = high / nyq
    # Clamp pour éviter erreur numérique
    low_n  = max(low_n,  0.001)
    high_n = min(high_n, 0.999)
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal)

# Filtre passe-bas léger uniquement pour visualisation
def lowpass_viz(signal, cutoff=10, fs=64, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, min(cutoff/nyq, 0.999), btype='low')
    return filtfilt(b, a, signal)

def extract_features(window: np.ndarray, fs: int = 64) -> np.ndarray:
    """
    Extraction des features sur signal filtré passe-bande.
    Le Freeze Index est maintenant calculé sur un signal
    qui contient réellement de l'énergie dans 3–8 Hz.
    """
    features = []
    for ax in range(window.shape[1]):
        s = window[:, ax]
        # Features temporelles
        features += [
            np.mean(s), np.std(s),
            np.max(s),  np.min(s),
            np.max(s) - np.min(s),
            np.sqrt(np.mean(s ** 2))   # RMS
        ]
        # Features fréquentielles
        fv = np.abs(np.fft.rfft(s))
        fr = np.fft.rfftfreq(len(s), d=1/fs)
        features.append(np.sum(fv))    # énergie totale

        # Freeze Index — rapport bande freeze / bande locomotion
        # Valide car le passe-bande préserve ces deux bandes
        fz = fv[(fr >= 3)  & (fr <= 8)]
        lc = fv[(fr >= 0.5) & (fr <= 3)]
        fi = np.sum(fz**2) / (np.sum(lc**2) + 1e-9)
        features.append(fi)

    return np.array(features)

def predict_win(window: np.ndarray, model, scaler):
    w = window.copy()
    # Appliquer le passe-bande AVANT extraction des features
    for i in range(w.shape[1]):
        w[:, i] = bandpass_filter(w[:, i])
    feat   = extract_features(w).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred   = model.predict(feat_s)[0]
    proba  = model.predict_proba(feat_s)[0][1]
    return int(pred), round(float(proba) * 100, 1)

# ============================================
# SCORE DE RISQUE
# ============================================
def calc_score(cadence, variab, fi, prob_fog, n_fog, duree):
    s = 0
    if cadence < 90:   s += 20
    elif cadence < 100: s += 10
    if variab > 0.5:   s += 25
    elif variab > 0.2: s += 12
    if fi > 3.0:       s += 20
    elif fi > 1.5:     s += 10
    s += int(prob_fog * 0.20)
    if duree > 0:
        freq = (n_fog / duree) * 60
        if freq > 0.5:  s += 15
        elif freq > 0.2: s += 8
    return min(int(s), 100)

def risk_level(score):
    if score < 30:  return "Faible", "low",  "#27AE60"
    if score < 65:  return "Modéré", "med",  "#E67E22"
    return "Élevé",  "high", "#C0392B"

# ============================================
# INTERPRÉTATION CLINIQUE
# ============================================
def interpret_signal(cadence, variab, fi, fog_pred,
                     proba_moy, score, labels_real):
    n_fog_real = sum(labels_real)
    fog_pct    = (n_fog_real / max(len(labels_real), 1)) * 100
    anomalies  = []

    if cadence < 90:
        anomalies.append(
            f"La cadence est réduite à <b>{cadence:.0f} pas/min</b> "
            "(valeur de référence ≈ 110 ppm). Sur la courbe, "
            "cela se traduit par un espacement élargi et irrégulier "
            "entre les impacts successifs du talon au sol, visibles "
            "comme des pics d'accélération plus espacés."
        )
    if variab > 0.2:
        anomalies.append(
            f"La variabilité inter-foulée est élevée "
            f"({variab:.3f} s ; seuil pathologique &gt; 0.05 s). "
            "Sur la courbe temporelle, les cycles de marche présentent "
            "des longueurs inégales — certains nettement plus courts "
            "ou plus longs que la moyenne. Cette irrégularité est un "
            "marqueur précoce d'instabilité locomotrice."
        )
    if fi > 1.5:
        anomalies.append(
            f"Le Freeze Index atteint <b>{fi:.2f}</b> (seuil ≥ 1.5 = "
            "pathologique). Ce biomarqueur exprime le rapport entre "
            "l'énergie spectrale dans la bande de freeze (3–8 Hz) et "
            "la bande de locomotion (0.5–3 Hz). Dans les zones rouges "
            "de la courbe, l'amplitude chute quasi à zéro : le patient "
            "est en état de gel de la marche — ses pieds ne quittent "
            "plus le sol malgré la volonté de marcher."
        )
    if fog_pct > 10:
        anomalies.append(
            f"Les <b>zones surlignées en rouge</b> sur la courbe "
            f"couvrent {fog_pct:.1f}% de la session "
            f"({n_fog_real} fenêtres de 2 s chacune). "
            "Chaque zone correspond à un épisode de Freezing of Gait "
            "annotés par des cliniciens dans le jeu de données."
        )

    if score < 30:
        conclusion = (
            "Le profil cinématique de marche est dans les limites "
            "normales attendues pour un patient parkinsonien stable. "
            "Les paramètres temporels, fréquentiels et la variabilité "
            "sont cohérents avec une démarche bien compensée. "
            "Un suivi périodique de routine est suffisant."
        )
    elif score < 65:
        conclusion = (
            "Plusieurs indicateurs convergent vers une instabilité "
            "locomotrice modérée. La réduction de cadence et "
            "l'élévation de la variabilité suggèrent une dégradation "
            "progressive du contrôle moteur. Un ajustement du "
            "protocole thérapeutique et une consultation neurologique "
            "sont conseillés."
        )
    else:
        conclusion = (
            "Le profil de marche est cliniquement préoccupant. "
            "La présence simultanée d'un Freeze Index élevé, "
            "d'une forte variabilité et d'épisodes FoG fréquents "
            "indique un risque de chute immédiat. Une consultation "
            "neurologique urgente est fortement recommandée, "
            "avec révision du traitement dopaminergique."
        )
    return anomalies, conclusion

# ============================================
# RAPPORT PDF
# ============================================
def gen_pdf(patient, age, sexe, date, duree, n_fog, n_normal,
            score, cadence, variab, fi, user):
    pdf = FPDF(); pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    niveau, _, _ = risk_level(score)

    # En-tête
    pdf.set_fill_color(13, 33, 55); pdf.rect(0, 0, 210, 36, 'F')
    pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(15, 10); pdf.cell(0, 8, "NeuroGait — Rapport d'Analyse", ln=True)
    pdf.set_font("Helvetica", "", 9); pdf.set_xy(15, 21)
    pdf.cell(0, 6, "Détection du Freezing of Gait — PFE Ingénierie Biomédicale 2026")

    pdf.set_xy(15, 44)
    pdf.set_text_color(13, 33, 55); pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Dossier Patient", ln=True)
    pdf.set_draw_color(46, 134, 193); pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y()); pdf.ln(3)
    for lbl, val in [("Patient", patient),("Âge", f"{age} ans"),
                     ("Sexe", sexe),("Date", date),("Utilisateur", user)]:
        pdf.set_font("Helvetica","B",9); pdf.cell(55,5.5,f"{lbl} :",ln=False)
        pdf.set_font("Helvetica","",9); pdf.cell(0,5.5,val,ln=True)

    pdf.ln(4); pdf.set_font("Helvetica","B",11)
    pdf.cell(0,7,"Score de Risque",ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)
    pdf.set_fill_color(225,232,240); pdf.rect(15,pdf.get_y(),180,9,'F')
    if score<30:   pdf.set_fill_color(26,86,50)
    elif score<65: pdf.set_fill_color(120,66,18)
    else:          pdf.set_fill_color(100,30,22)
    pdf.rect(15,pdf.get_y(),int(180*score/100),9,'F')
    pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",9)
    pdf.set_xy(15,pdf.get_y()+1)
    pdf.cell(180,7,f"  {score}/100 — Risque {niveau}",ln=True)
    pdf.ln(2); pdf.set_text_color(13,33,55); pdf.set_font("Helvetica","",9)
    if score<30:
        pdf.multi_cell(0,5,"Profil de marche dans les limites normales. Surveillance de routine.")
    elif score<65:
        pdf.multi_cell(0,5,"Instabilité modérée détectée. Consultation neurologique conseillée.")
    else:
        pdf.multi_cell(0,5,"RISQUE ÉLEVÉ. Consultation neurologique urgente recommandée.")

    pdf.ln(4); pdf.set_font("Helvetica","B",11)
    pdf.cell(0,7,"Biomarqueurs",ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)
    hdrs=["Paramètre","Valeur","Référence","Statut"]; cw=[68,35,47,35]
    pdf.set_fill_color(13,33,55); pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",9)
    for i,h in enumerate(hdrs): pdf.cell(cw[i],7,h,border=1,fill=True)
    pdf.ln()
    rows=[("Durée session",f"{duree:.0f} s","—","—"),
          ("Cadence",f"{cadence:.1f} ppm","~110 ppm",
           "Anormal" if cadence<90 else "Normal"),
          ("Variabilité",f"{variab:.4f} s","< 0.05 s",
           "Anormal" if variab>0.2 else "Normal"),
          ("Freeze Index",f"{fi:.3f}","< 1.5",
           "Anormal" if fi>1.5 else "Normal"),
          ("Épisodes FoG",str(n_fog),"0",
           "Présent" if n_fog>0 else "Absent"),
          ("Fenêtres normales",str(n_normal),"—","—")]
    pdf.set_font("Helvetica","",9)
    for i,row in enumerate(rows):
        if i%2==0: pdf.set_fill_color(247,249,252)
        else: pdf.set_fill_color(255,255,255)
        pdf.set_text_color(13,33,55)
        for j,c in enumerate(row):
            if j==3 and c in ["Anormal","Présent"]:
                pdf.set_text_color(146,43,33)
            elif j==3: pdf.set_text_color(26,86,50)
            pdf.cell(cw[j],6,c,border=1,fill=True)
            pdf.set_text_color(13,33,55)
        pdf.ln()

    pdf.set_y(-27); pdf.set_draw_color(46,134,193)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)
    pdf.set_font("Helvetica","",8); pdf.set_text_color(150,160,170)
    pdf.cell(0,5,f"Généré le {date} — NeuroGait · PFE Ingénierie Biomédicale 2026",
             ln=True, align='C')
    pdf.cell(0,5,"Ce document est généré automatiquement. Il ne remplace pas un diagnostic médical.",
             ln=True, align='C')
    return bytes(pdf.output())

# ============================================
# GMAIL
# ============================================
def send_gmail(dest, patient, n_fog, score, gu, gp, pdf_b=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = gu; msg['To'] = dest
        niveau, _, _ = risk_level(score)
        msg['Subject'] = f"Alerte NeuroGait — {patient} · Score {score}/100"
        body = (f"Bonjour,\n\nLe système NeuroGait signale une activité "
                f"de marche nécessitant votre attention.\n\n"
                f"Patient      : {patient}\n"
                f"Date         : {datetime.now().strftime('%d/%m/%Y à %H:%M')}\n"
                f"Épisodes FoG : {n_fog}\n"
                f"Score        : {score}/100 — Risque {niveau}\n\n"
                f"Le rapport PDF est joint à ce message.\n\n"
                f"Cordialement,\nSystème NeuroGait — PFE Ingénierie Biomédicale")
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        if pdf_b:
            p = MIMEBase('application', 'octet-stream')
            p.set_payload(pdf_b); encoders.encode_base64(p)
            p.add_header('Content-Disposition',
                         f'attachment; filename="rapport_{patient}.pdf"')
            msg.attach(p)
        srv = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        srv.login(gu, gp); srv.sendmail(gu, dest, msg.as_string())
        srv.quit()
        return True, "Email envoyé avec succès."
    except Exception as e:
        return False, str(e)

# ============================================
# PAGE LOGIN
# ============================================
if not st.session_state['authenticated']:
    st.markdown("""<style>
    [data-testid="stSidebar"]{display:none!important;}
    .block-container{padding:0!important;max-width:100%!important;}
    </style>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-bottom:6px;font-size:2.2rem;'>🔬</div>
        <div style='font-family:Libre Baskerville,serif;font-size:1.85rem;
                    font-weight:700;color:#0D2137;text-align:center;margin-bottom:3px;'>
            NeuroGait
        </div>
        <div style='font-size:.82rem;color:#5D7A8A;text-align:center;
                    margin-bottom:28px;line-height:1.55;'>
            Système d'Analyse de la Marche Parkinsonienne<br>
            PFE — Ingénierie Biomédicale
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("""
            <div style='background:white;border-radius:10px;padding:30px 28px 24px;
                        box-shadow:0 4px 24px rgba(13,33,55,0.11);
                        border:1px solid #D6E4F0;'>
            </div>""", unsafe_allow_html=True)
            username = st.text_input("Identifiant",
                                     placeholder="Entrez votre identifiant")
            password = st.text_input("Mot de passe", type="password",
                                     placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Connexion", use_container_width=True):
                if check_password(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Identifiant ou mot de passe incorrect.")

        st.markdown("""
        <div style='text-align:center;margin-top:18px;font-size:.73rem;
                    color:#B0BEC5;line-height:1.7;'>
            Accès réservé au personnel médical autorisé<br>© 2026 NeuroGait
        </div>""", unsafe_allow_html=True)

        with st.expander("Comptes de démonstration"):
            st.markdown("""
| Identifiant | Mot de passe |
|---|---|
| `admin` | `admin123` |
| `medecin` | `parkinson2026` |
| `lina` | `pfe2026` |""")
    st.stop()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown(f"""
    <div style='padding:22px 0 14px;text-align:center;'>
        <div style='font-size:2rem;'>🔬</div>
        <div style='font-family:Libre Baskerville,serif;font-size:1.1rem;
                    font-weight:700;color:white;margin-top:5px;'>NeuroGait</div>
        <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                    letter-spacing:2px;text-transform:uppercase;'>
            Analyse de Marche
        </div>
    </div><hr>
    <div style='background:rgba(255,255,255,.07);border-radius:6px;
                padding:8px 12px;margin-bottom:14px;'>
        <div style='font-size:.62rem;color:rgba(255,255,255,.38);
                    text-transform:uppercase;letter-spacing:1px;'>Session</div>
        <div style='font-size:.88rem;font-weight:600;color:white;margin-top:2px;'>
            {st.session_state['username']}
        </div>
    </div>
    <div class='sec-lbl' style='color:rgba(255,255,255,.32);
         border-color:rgba(255,255,255,.07);'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["Analyse Fichier", "Démo Temps Réel",
                         "Historique", "Configuration"],
                    label_visibility="collapsed")

    st.markdown("""<hr><div class='sec-lbl'
    style='color:rgba(255,255,255,.32);border-color:rgba(255,255,255,.07);'>
    Dossier Patient</div>""", unsafe_allow_html=True)

    nom  = st.text_input("Nom", "Patient 001", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1: age  = st.number_input("Âge",  0, 120, 65, label_visibility="collapsed")
    with c2: sexe = st.selectbox("Sexe", ["H","F"], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    if model_ok:
        st.markdown("<div style='background:rgba(26,86,50,.22);border:1px solid "
                    "rgba(39,174,96,.28);border-radius:5px;padding:7px 11px;"
                    "font-size:.76rem;color:#7DCEA0;font-weight:600;'>"
                    "Modèle ML chargé</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:rgba(146,43,33,.2);border:1px solid "
                    "rgba(192,57,43,.28);border-radius:5px;padding:7px 11px;"
                    "font-size:.76rem;color:#F1948A;font-weight:600;'>"
                    "Modèle non trouvé</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Déconnexion", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ''
        st.rerun()

# ============================================
# HEADER
# ============================================
st.markdown(f"""
<div class="page-header">
    <div class="ph-user">{st.session_state['username']}</div>
    <div class="ph-eyebrow">Système de Diagnostic Médical</div>
    <div class="ph-title">Analyse de la Marche Parkinsonienne</div>
    <div class="ph-sub">
        Détection FoG · Score de Risque · Interprétation Clinique · Rapport PDF
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE — ANALYSE FICHIER
# ============================================
if page == "Analyse Fichier":

    st.markdown('<div class="sec-lbl">Chargement des données</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("Fichier .txt (Daphnet) ou .csv",
                                type=["txt","csv"],
                                label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div style='background:white;border-radius:10px;padding:52px;
                    text-align:center;border:1px solid #D6E4F0;'>
            <div style='font-size:2.6rem;margin-bottom:12px;'>📂</div>
            <div style='font-family:Libre Baskerville,serif;font-size:1.2rem;
                        font-weight:700;color:#0D2137;margin-bottom:7px;'>
                Aucun fichier chargé
            </div>
            <div style='color:#5D7A8A;font-size:.86rem;max-width:320px;
                        margin:0 auto;line-height:1.6;'>
                Glissez un fichier <b>.txt</b> du dataset Daphnet FoG
                pour démarrer l'analyse.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    col_names = ['time','ankle_x','ankle_y','ankle_z',
                 'thigh_x','thigh_y','thigh_z',
                 'trunk_x','trunk_y','trunk_z','label']
    try:
        df = pd.read_csv(uploaded, sep=" ", header=None, names=col_names)
        df = df[df['label'] != 0].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erreur de lecture : {e}"); st.stop()

    # Pipeline
    WS, OL = 128, 64
    windows_t, labels_pred, labels_real, probas = [], [], [], []

    for start in range(0, len(df) - WS, OL):
        end = start + WS
        win = df[['ankle_x','ankle_y','ankle_z']].iloc[start:end].values
        lbl = df['label'].iloc[start:end].mode()[0]
        if model_ok:
            pred, prob = predict_win(win, model, scaler)
            labels_pred.append(pred); probas.append(prob)
        labels_real.append(1 if lbl == 2 else 0)
        windows_t.append(start / 64)

    fog_pred  = sum(labels_pred) if model_ok else 0
    fog_real  = sum(labels_real)
    duree     = len(df) / 64
    az        = df['ankle_z'].values

    # Biomarqueurs sur signal passe-bande
    az_bp = bandpass_filter(az)
    try:
        peaks, _ = find_peaks(az_bp, distance=32)
        if len(peaks) > 2:
            diffs   = np.diff(peaks) / 64
            cadence = float(60 / np.mean(diffs))
            variab  = float(np.std(diffs))
        else:
            cadence, variab = 100.0, 0.1
    except Exception:
        cadence, variab = 100.0, 0.1
    cadence = min(max(cadence, 50), 150)
    variab  = min(variab, 2.0)

    N     = min(512, len(az_bp))
    fv    = np.abs(np.fft.rfft(az_bp[:N]))
    fr    = np.fft.rfftfreq(N, d=1/64)
    fz    = fv[(fr >= 3) & (fr <= 8)]
    lc_   = fv[(fr >= 0.5) & (fr <= 3)]
    fi    = float(np.sum(fz**2) / (np.sum(lc_**2) + 1e-9))
    fi    = min(fi, 10.0)

    p_avg = float(np.mean(probas)) if probas else 0.0
    score = calc_score(cadence, variab, fi, p_avg/100, fog_pred, duree)
    niv, sc_cls, sc_col = risk_level(score)
    anomalies, conclusion = interpret_signal(cadence, variab, fi,
                                             fog_pred, p_avg, score,
                                             labels_real)

    # KPI
    cols = st.columns(5)
    kpis = [
        ("", "Durée session", f"{duree:.0f} s", f"{len(df):,} points"),
        ("ok" if fog_pred==0 else "dk",
         "Marche normale",
         str(len(labels_pred)-fog_pred if labels_pred else 0),
         "fenêtres saines"),
        ("dk", "Épisodes FoG", str(fog_pred), "détectés par IA"),
        ("wk" if cadence<100 else "ok",
         "Cadence", f"{cadence:.0f}", "pas / min"),
        ("dk" if score>=65 else ("wk" if score>=30 else "ok"),
         "Score de risque", str(score), f"Risque {niv}"),
    ]
    for col, (cls, lbl, val, note) in zip(cols, kpis):
        with col:
            st.markdown(f'<div class="kpi {cls}"><div class="kpi-lbl">{lbl}</div>'
                        f'<div class="kpi-val">{val}</div>'
                        f'<div class="kpi-note">{note}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_main, col_side = st.columns([3, 1])

    with col_main:
        # Signal annoté
        st.markdown('<div class="sec-lbl">Signal accéléromètre — cheville (axe Z)</div>',
                    unsafe_allow_html=True)

        az_viz = lowpass_viz(az)   # version lissée pour affichage uniquement
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=az_viz[:3000], mode='lines',
            name='Signal (lissé — visualisation)',
            line=dict(color='#1A5276', width=1.3),
            hovertemplate='%{y:.1f} mg<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            y=az[:3000], mode='lines', name='Signal brut',
            line=dict(color='#AED6F1', width=0.6, dash='dot'),
            opacity=0.4,
            hovertemplate='Brut : %{y:.1f} mg<extra></extra>'
        ))

        in_fog = False; sf = 0; n_annot = 0
        for i in range(min(3000, len(df))):
            if df['label'].iloc[i] == 2 and not in_fog:
                sf = i; in_fog = True
            elif df['label'].iloc[i] != 2 and in_fog:
                fig.add_vrect(x0=sf, x1=i, fillcolor="#C0392B",
                              opacity=0.10, line_width=0)
                if n_annot == 0:
                    fig.add_annotation(
                        x=(sf+i)//2, y=float(np.max(az_viz[:3000]))*0.95,
                        text="Épisode FoG<br>(gel de la marche)",
                        showarrow=True, arrowhead=2,
                        arrowcolor="#922B21", arrowsize=1.1,
                        font=dict(size=9, color="#922B21"),
                        ax=0, ay=-32
                    )
                n_annot += 1; in_fog = False

        if len(az_viz[:3000]) > 60:
            fig.add_annotation(
                x=25, y=float(np.percentile(az_viz[:3000], 20)),
                text="Marche normale :<br>pics réguliers",
                showarrow=False,
                font=dict(size=9, color="#1A5276"),
                bgcolor="rgba(235,245,251,.88)",
                bordercolor="#AED6F1", borderwidth=1, borderpad=4
            )

        fig.update_layout(
            height=295,
            margin=dict(l=5, r=5, t=5, b=5),
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(color='#1A252F', family='Source Sans 3', size=11),
            xaxis=dict(title="Échantillons", showgrid=True,
                       gridcolor='#F0F4F8', linecolor='#D6E4F0',
                       tickfont=dict(size=10), zeroline=False),
            yaxis=dict(title="mg", showgrid=True,
                       gridcolor='#F0F4F8', linecolor='#D6E4F0',
                       tickfont=dict(size=10), zeroline=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.01,
                        font=dict(size=10), bgcolor='rgba(0,0,0,0)',
                        borderwidth=0),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={'displayModeBar': False,
                                'responsive': True})

        # Courbe de probabilité
        if model_ok and probas:
            st.markdown('<div class="sec-lbl">Probabilité FoG au fil du temps</div>',
                        unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=windows_t, y=probas, mode='lines',
                fill='tozeroy',
                line=dict(color='#1A5276', width=1.4),
                fillcolor='rgba(26,82,118,.07)',
                hovertemplate='t=%{x:.0f}s  P(FoG)=%{y:.1f}%<extra></extra>'
            ))
            fig2.add_hrect(y0=60, y1=105,
                           fillcolor="rgba(192,57,43,.05)", line_width=0)
            fig2.add_hline(y=60, line_dash="dash",
                           line_color="#C0392B", line_width=1,
                           annotation_text="Seuil 60 %",
                           annotation_font=dict(size=9, color="#922B21"))
            fig2.update_layout(
                height=175,
                margin=dict(l=5, r=5, t=5, b=5),
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(color='#1A252F', family='Source Sans 3', size=11),
                xaxis=dict(title="Temps (s)", showgrid=True,
                           gridcolor='#F0F4F8', zeroline=False,
                           tickfont=dict(size=10)),
                yaxis=dict(title="%", range=[0, 105], showgrid=True,
                           gridcolor='#F0F4F8', zeroline=False,
                           tickfont=dict(size=10)),
                showlegend=False, hovermode='x unified'
            )
            st.plotly_chart(fig2, use_container_width=True,
                            config={'displayModeBar': False,
                                    'responsive': True})

    with col_side:
        # Score
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <div class="card-title">Score de risque</div>
            <div class="score-ring {sc_cls}">
                <div class="score-num">{score}</div>
                <div class="score-denom">/100</div>
            </div>
            <div class="score-title">Risque {niv}</div>
            <div class="score-sub">{datetime.now().strftime("%d/%m/%Y")}</div>
        </div>""", unsafe_allow_html=True)

        # Features
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Biomarqueurs</div>',
                    unsafe_allow_html=True)
        tag_l = {"ok": "Normal", "med": "Modéré", "nok": "Anormal"}
        feat_rows = [
            ("Cadence", f"{cadence:.1f} ppm", "~110 ppm",
             "ok" if cadence>=100 else ("med" if cadence>=85 else "nok")),
            ("Variabilité", f"{variab:.4f} s", "< 0.05 s",
             "ok" if variab<0.05 else ("med" if variab<0.2 else "nok")),
            ("Freeze Index", f"{fi:.3f}", "< 1.5",
             "ok" if fi<1.5 else ("med" if fi<3.0 else "nok")),
            ("P(FoG) moy.", f"{p_avg:.1f}%", "< 30%",
             "ok" if p_avg<30 else ("med" if p_avg<60 else "nok")),
            ("FoG détectés", str(fog_pred), "0",
             "ok" if fog_pred==0 else ("med" if fog_pred<=3 else "nok")),
        ]
        for fn, fv_, fr_, ft in feat_rows:
            st.markdown(f"""
            <div class="feat-row">
                <div>
                    <div class="feat-name">{fn}</div>
                    <div class="feat-ref">Réf : {fr_}</div>
                </div>
                <div style="text-align:right;">
                    <div class="feat-val">{fv_}</div>
                    <span class="tag {ft}">{tag_l[ft]}</span>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Interprétation
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Interprétation clinique du signal</div>',
                unsafe_allow_html=True)
    if anomalies:
        for i, a in enumerate(anomalies):
            crit = "FoG" in a or "Freeze" in a or "rouge" in a
            css = "crit" if crit else ""
            st.markdown(f"""
            <div class="interp-box {css}">
                <div class="interp-title {css}">Observation {i+1}</div>
                {a}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="interp-box">
            <div class="interp-title">Observations</div>
            Le signal présente un profil rythmique régulier. Les pics
            d'accélération sont espacés uniformément, traduisant un
            cycle de marche stable, sans anomalie notable.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box {'crit' if score>=65 else ''}">
        <div class="interp-title {'crit' if score>=65 else ''}">
            Conclusion générale
        </div>
        {conclusion}
    </div>""", unsafe_allow_html=True)

    # Actions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Actions</div>', unsafe_allow_html=True)
    pdf_b = gen_pdf(nom, age, sexe,
                    datetime.now().strftime("%d/%m/%Y %H:%M"),
                    duree, fog_pred,
                    len(labels_pred)-fog_pred if labels_pred else 0,
                    score, cadence, variab, fi,
                    st.session_state['username'])
    a1, a2 = st.columns(2)
    with a1:
        st.download_button(
            "Télécharger le rapport PDF", data=pdf_b,
            file_name=f"rapport_{nom}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf", use_container_width=True)
    with a2:
        if st.button("Sauvegarder dans l'historique",
                     use_container_width=True):
            save_session(nom, age, sexe, duree, fog_pred,
                         len(labels_pred)-fog_pred if labels_pred else 0,
                         score, cadence, variab, fi,
                         st.session_state['username'])
            st.success("Session enregistrée.")
    if score >= 65 and fog_pred > 0:
        st.markdown(f"""
        <div class="alert-crit">
            <b>Risque élevé détecté</b>
            <small>{fog_pred} épisodes FoG · Score {score}/100 ·
            Configurez les alertes email dans l'onglet Configuration.</small>
        </div>""", unsafe_allow_html=True)

# ============================================
# PAGE — DÉMO TEMPS RÉEL
# CORRECTION 2 : streaming de données Daphnet
# réelles au lieu de bruit aléatoire
# ============================================
elif page == "Démo Temps Réel":

    st.markdown("""
    <div class="interp-box" style="margin-bottom:16px;">
        <div class="interp-title">Mode de simulation</div>
        Cette démonstration injecte les données d'un patient réel
        du dataset Daphnet FoG ligne par ligne, reproduisant
        fidèlement les conditions d'acquisition en temps réel.
        Chargez un fichier pour démarrer.
    </div>""", unsafe_allow_html=True)

    demo_file = st.file_uploader(
        "Fichier patient pour la simulation (format Daphnet .txt)",
        type=["txt"], key="demo_upload",
        label_visibility="collapsed"
    )

    if demo_file is not None:
        col_names = ['time','ankle_x','ankle_y','ankle_z',
                     'thigh_x','thigh_y','thigh_z',
                     'trunk_x','trunk_y','trunk_z','label']
        df_demo = pd.read_csv(demo_file, sep=" ",
                              header=None, names=col_names)
        df_demo = df_demo[df_demo['label'] != 0].reset_index(drop=True)
        st.session_state['demo_df'] = df_demo
        st.success(f"Fichier chargé : {len(df_demo)} échantillons · "
                   f"{(df_demo['label']==2).sum()} points FoG annotés")

    col_c, col_d = st.columns([3, 1])
    with col_d:
        st.markdown('<div class="sec-lbl">Diagnostic</div>',
                    unsafe_allow_html=True)
        dx_box  = st.empty()
        pb_box  = st.empty()
        st_box  = st.empty()
        dx_box.markdown('<div class="dx wait"><div class="dx-icon">◌</div>'
                        '<div class="dx-title">En attente</div>'
                        '<div class="dx-sub">Démarrez la session</div></div>',
                        unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="sec-lbl">Signal IMU — temps réel</div>',
                    unsafe_allow_html=True)
        ch_box = st.empty()

    c1, c2, _ = st.columns([1, 1, 5])
    with c1:
        s1 = st.button("Démarrer", type="primary",
                       use_container_width=True)
    with c2:
        s2 = st.button("Arrêter", use_container_width=True)
    if s1: st.session_state['running'] = True
    if s2: st.session_state['running'] = False

    if st.session_state.get('running', False):
        df_src = st.session_state.get('demo_df', None)
        if df_src is None:
            st.warning("Chargez d'abord un fichier patient ci-dessus.")
            st.session_state['running'] = False
        else:
            buf = collections.deque(maxlen=128)
            hx, hy, hz, hlbl = [], [], [], []
            nf = nn = 0; lp = 0; lpr = 0.0
            idx = st.session_state.get('demo_idx', 0)

            while st.session_state.get('running', False):
                if idx >= len(df_src):
                    idx = 0   # boucle sur le fichier

                row = df_src.iloc[idx]
                ax_, ay_, az_ = (float(row['ankle_x']),
                                 float(row['ankle_y']),
                                 float(row['ankle_z']))
                real_lbl = int(row['label'])

                buf.append([ax_, ay_, az_])
                hx.append(ax_); hy.append(ay_)
                hz.append(az_); hlbl.append(real_lbl)

                N_SHOW = 320
                if len(hx) > N_SHOW:
                    hx=hx[-N_SHOW:]; hy=hy[-N_SHOW:]
                    hz=hz[-N_SHOW:]; hlbl=hlbl[-N_SHOW:]

                if len(buf) == 128 and model_ok:
                    lp, lpr = predict_win(np.array(buf), model, scaler)
                    if lp == 1: nf += 1
                    else: nn += 1

                # Diagnostic
                if lp == 1:
                    dx_box.markdown(
                        '<div class="dx fog"><div class="dx-icon">●</div>'
                        '<div class="dx-title">FoG détecté</div>'
                        '<div class="dx-sub">Enrayement cinétique</div>'
                        '</div>', unsafe_allow_html=True)
                else:
                    dx_box.markdown(
                        '<div class="dx normal"><div class="dx-icon">●</div>'
                        '<div class="dx-title">Marche normale</div>'
                        '<div class="dx-sub">Aucune anomalie</div>'
                        '</div>', unsafe_allow_html=True)

                bc = "#C0392B" if lpr > 60 else "#1A5276"
                pb_box.markdown(f"""
                <div class="pb-wrap">
                    <div class="pb-lbl">Probabilité FoG</div>
                    <div class="pb-bg">
                        <div class="pb-fill"
                             style="width:{lpr}%;background:{bc};"></div>
                    </div>
                    <div class="pb-num" style="color:{bc};">{lpr}%</div>
                </div>""", unsafe_allow_html=True)

                st_box.markdown(f"""
                <div class="card">
                    <div class="feat-row">
                        <span class="feat-name">Normal</span>
                        <span class="feat-val" style="color:#1A5632;">{nn}</span>
                    </div>
                    <div class="feat-row">
                        <span class="feat-name">FoG</span>
                        <span class="feat-val" style="color:#922B21;">{nf}</span>
                    </div>
                    <div class="feat-row" style="border:none;">
                        <span class="feat-name" style="font-size:.75rem;
                              color:var(--muted);">Label réel</span>
                        <span class="tag {'nok' if real_lbl==2 else 'ok'}">
                            {'FoG' if real_lbl==2 else 'Normal'}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Graphique
                is_fog_now = (real_lbl == 2)
                lc_ = '#C0392B' if is_fog_now else '#1A5276'
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hx, mode='lines', name='X',
                    line=dict(color='#AED6F1', width=0.9)))
                fig.add_trace(go.Scatter(y=hy, mode='lines', name='Y',
                    line=dict(color='#A9DFBF', width=0.9)))
                fig.add_trace(go.Scatter(y=hz, mode='lines', name='Z',
                    line=dict(color=lc_, width=1.7)))
                fig.update_layout(
                    height=305,
                    margin=dict(l=5, r=5, t=5, b=5),
                    plot_bgcolor='white', paper_bgcolor='white',
                    font=dict(color='#1A252F',
                              family='Source Sans 3', size=11),
                    xaxis=dict(showgrid=True, gridcolor='#F0F4F8',
                               zeroline=False, tickfont=dict(size=10)),
                    yaxis=dict(title='mg', showgrid=True,
                               gridcolor='#F0F4F8', zeroline=False,
                               tickfont=dict(size=10)),
                    legend=dict(orientation='h', yanchor='bottom',
                                y=1.01, font=dict(size=10),
                                bgcolor='rgba(0,0,0,0)', borderwidth=0)
                )
                ch_box.plotly_chart(fig, use_container_width=True,
                                    config={'displayModeBar': False,
                                            'responsive': True})
                idx += 1
                st.session_state['demo_idx'] = idx
                time.sleep(1/64)   # respect de la fréquence 64 Hz

    elif not st.session_state.get('demo_df') is None:
        with col_c:
            st.markdown("""
            <div style='background:white;border-radius:9px;padding:44px;
                        text-align:center;border:1px solid #D6E4F0;'>
                <div style='font-size:2.4rem;margin-bottom:10px;'>▶</div>
                <div style='font-family:Libre Baskerville,serif;font-size:1.1rem;
                            font-weight:700;color:#0D2137;'>
                    Cliquez sur Démarrer
                </div>
            </div>""", unsafe_allow_html=True)

# ============================================
# PAGE — HISTORIQUE
# ============================================
elif page == "Historique":

    st.markdown('<div class="sec-lbl">Historique des sessions</div>',
                unsafe_allow_html=True)
    try:
        df_h = get_sessions()
        if df_h.empty:
            st.info("Aucune session. Analysez un fichier et "
                    "cliquez sur Sauvegarder.")
        else:
            cols = st.columns(4)
            for col, cls, lbl, val, note in [
                (cols[0], "",   "Sessions",
                 len(df_h), "enregistrées"),
                (cols[1], "dk", "FoG total",
                 int(df_h['fog_episodes'].sum()), "épisodes"),
                (cols[2], "wk", "Score moyen",
                 f"{df_h['score_risque'].mean():.0f}", "/100"),
                (cols[3], "ok", "Patients",
                 df_h['patient'].nunique(), "uniques"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="kpi {cls}"><div class="kpi-lbl">'
                        f'{lbl}</div><div class="kpi-val">{val}</div>'
                        f'<div class="kpi-note">{note}</div></div>',
                        unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-lbl">Évolution du score de risque</div>',
                        unsafe_allow_html=True)

            fig_h = go.Figure()
            for pt in df_h['patient'].unique():
                dp = df_h[df_h['patient'] == pt]
                fig_h.add_trace(go.Scatter(
                    x=dp['date'], y=dp['score_risque'],
                    mode='lines+markers', name=pt,
                    line=dict(width=2)))
            fig_h.add_hline(y=65, line_dash="dash",
                            line_color="#C0392B", line_width=1,
                            annotation_text="Seuil risque élevé",
                            annotation_font=dict(size=9, color="#922B21"))
            fig_h.update_layout(
                height=250,
                plot_bgcolor='white', paper_bgcolor='white',
                font=dict(color='#1A252F', family='Source Sans 3', size=11),
                xaxis=dict(title="Date", showgrid=True,
                           gridcolor='#F0F4F8', zeroline=False),
                yaxis=dict(title="Score", range=[0, 105],
                           showgrid=True, gridcolor='#F0F4F8',
                           zeroline=False),
                margin=dict(l=5, r=5, t=5, b=5))
            st.plotly_chart(fig_h, use_container_width=True,
                            config={'displayModeBar': False})

            st.markdown('<div class="sec-lbl">Détail</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                df_h[['patient','age','date','duree',
                      'fog_episodes','score_risque','utilisateur']]
                .rename(columns={
                    'patient':'Patient','age':'Âge','date':'Date',
                    'duree':'Durée (s)','fog_episodes':'FoG',
                    'score_risque':'Score','utilisateur':'Utilisateur'}),
                use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

# ============================================
# PAGE — CONFIGURATION
# ============================================
elif page == "Configuration":

    st.markdown('<div class="sec-lbl">Alertes email (Gmail)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="interp-box">
        <div class="interp-title">Configuration requise</div>
        Activez la validation en deux étapes sur votre compte Gmail,
        puis générez un <b>mot de passe d'application</b> dans
        Paramètres → Sécurité → Mots de passe des applications.
        Utilisez ce code de 16 caractères ci-dessous — et non votre
        mot de passe habituel.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expéditeur**")
        gu = st.text_input("Adresse Gmail",
                           placeholder="votre.email@gmail.com")
        gp = st.text_input("Mot de passe d'application",
                           type="password",
                           placeholder="xxxx xxxx xxxx xxxx")
    with c2:
        st.markdown("**Destinataire**")
        de = st.text_input("Email médecin",
                           placeholder="medecin@hopital.ma")
        st.slider("Alerte si FoG ≥", 1, 10, 3)
        st.slider("Alerte si Score ≥", 30, 100, 65)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Envoyer un email de test"):
        if not gu or not gp or not de:
            st.error("Remplissez tous les champs.")
        else:
            pb_ = gen_pdf("Patient Test", 65, "H",
                          datetime.now().strftime("%d/%m/%Y %H:%M"),
                          120, 5, 45, 72, 88.5, .45, 2.8,
                          st.session_state['username'])
            ok_, msg_ = send_gmail(de, "Patient Test", 5, 72, gu, gp, pb_)
            if ok_: st.success(msg_)
            else: st.error(msg_)

    st.markdown("---")
    st.markdown("**Envoi manuel d'une alerte**")
    m1, m2 = st.columns(2)
    with m1:
        pa = st.text_input("Patient", nom)
        fa = st.number_input("Épisodes FoG", 0, 100, 5)
    with m2:
        sa = st.number_input("Score de risque", 0, 100, 72)

    if st.button("Envoyer l'alerte", type="primary"):
        if not gu or not gp or not de:
            st.error("Configurez d'abord Gmail ci-dessus.")
        else:
            pb_ = gen_pdf(pa, age, sexe,
                          datetime.now().strftime("%d/%m/%Y %H:%M"),
                          0, fa, 0, sa, 100., .5, 2.,
                          st.session_state['username'])
            ok_, msg_ = send_gmail(de, pa, fa, sa, gu, gp, pb_)
            if ok_: st.success(msg_)
            else: st.error(msg_)
