# ==============================================================================
# app_cloud.py — NeuroGait (Version Complète et Finale PFE)
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

# ── CONFIGURATION DE PAGE ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroGait — Analyse de Marche",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── HABILITATIONS UTILISATEURS ────────────────────────────────────────────────
USERS = {
    "admin":   hashlib.sha256("admin123".encode()).hexdigest(),
    "medecin": hashlib.sha256("parkinson2026".encode()).hexdigest(),
    "lina":    hashlib.sha256("pfe2026".encode()).hexdigest(),
}

def check_password(u, p):
    return USERS.get(u) == hashlib.sha256(p.encode()).hexdigest()

# ── GESTION DE BASE DE DONNÉES SÉCURISÉE ──────────────────────────────────────
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

# ── STYLISATION CSS (Thème médical propre et contrasté) ───────────────────────
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

/* Forcer le conteneur en thème clair pour le rendu clinique */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--bg-grey) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    color: var(--text) !important;
}

/* Correction lisibilité formulaires */
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

/* Sidebar */
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

/* Composants UI */
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
.card-title {
    font-family: 'Libre Baskerville', serif; font-size: 1rem; font-weight: 700; color: var(--navy); margin-bottom: 12px;
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
.feat-name { font-weight: 500; }
.feat-val { font-weight: 600; font-family: 'Libre Baskerville', serif; }
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
.dx-icon { font-size: 1.8rem; margin-bottom: 4px; }
.dx-title { font-family: 'Libre Baskerville', serif; font-size: 1.1rem; font-weight: 700; }
.dx-sub { font-size: 0.8rem; opacity: 0.8; }

.pbar-wrap { background: white; border-radius: 8px; padding: 12px; border: 1px solid var(--border); margin-bottom: 12px; }
.pbar-lbl { font-size: 0.65rem; font-weight: 700; color: var(--muted); text-transform: uppercase; margin-bottom: 6px; }
.pbar-bg { background: #E8EDF2; border-radius: 4px; height: 8px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.pbar-num { font-family: 'Libre Baskerville', serif; font-size: 1.2rem; font-weight: 700; margin-top: 4px; }

.sec-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── GESTION DES SESSIONS ──────────────────────────────────────────────────────
for k, v in [('authenticated', False), ('username', ''), ('running', False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── CHARGEMENT MODÈLE ML & TRAITEMENT SIGNAL ──────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("models/model_fog_rf.pkl"), joblib.load("models/scaler_fog.pkl")

try:
    model, scaler = load_model()
    model_ok = True
except Exception:
    model_ok = False

def bw_filter(sig, cutoff=3.0, fs=64, order=4):
    """Filtre passe-bas pour lissage visuel et calcul de cadence."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, sig)

def bp_filter(sig, lowcut=0.5, highcut=20.0, fs=64, order=4):
    """Filtre passe-bande dédié à l'extraction ML (Préserve la bande FoG 3-8 Hz)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, sig)

def extract_features(win, fs=64):
    feats = []
    for ax in range(win.shape[1]):
        s = win[:, ax]
        feats += [np.mean(s), np.std(s), np.max(s), np.min(s),
                  np.max(s) - np.min(s), np.sqrt(np.mean(s**2))]
        
        fv = np.abs(np.fft.rfft(s))
        fr = np.fft.rfftfreq(len(s), d=1/fs)
        feats.append(np.sum(fv))
        
        # Freeze Index : Puissance(3-8 Hz) / Puissance(0.5-3 Hz)
        fz = fv[(fr >= 3.0) & (fr <= 8.0)]
        lc = fv[(fr >= 0.5) & (fr <= 3.0)]
        fi = np.sum(fz**2) / (np.sum(lc**2) + 1e-6)
        feats.append(fi)
    return np.array(feats)

def predict_win(win, model, scaler):
    w = win.copy()
    for i in range(3): 
        w[:, i] = bp_filter(w[:, i])
    f = extract_features(w).reshape(1, -1)
    fs = scaler.transform(f)
    pred = model.predict(fs)[0]
    prob = model.predict_proba(fs)[0][1]
    return int(pred), round(float(prob) * 100, 1)

# ── LOGIQUE DE SCORING MÉDICAL ────────────────────────────────────────────────
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
        anomalies.append(f"Forte variabilité inter-foulée ({variab:.3f} s). Instabilité locomotrice marquée.")
    if fi > 1.5:
        anomalies.append(f"Freeze Index critique ({fi:.2f} > seuil 1.5). Hypersynchronie spectrale dans la bande 3-8 Hz caractéristique d'un blocage moteur.")
    if fog_pct > 5:
        anomalies.append(f"Présence confirmée d'épisodes FoG ({fog_pct:.1f}% du tracé, {n_fog_real} fenêtres critiques).")

    if score < 30:
        concl = "Marche parkinsonienne stable et compensée. Poursuite du protocole actuel."
    elif score < 65:
        concl = "Dégradation modérée du contrôle locomoteur. Ajustement thérapeutique à envisager."
    else:
        concl = "Risque de chute imminent lié à des enrayements cinétiques majeurs. Évaluation neurologique prioritaire."
    return anomalies, concl

# ── GÉNÉRATION RAPPORT PDF (Encodage ASCII propre) ────────────────────────────
def gen_pdf(patient, age, sexe, date, duree, n_fog, n_normal, score, cadence, variab, fi, user):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    niveau, _,
