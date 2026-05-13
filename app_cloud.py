# ============================================
# app_cloud.py — NeuroGait
# Système d'Analyse de la Marche Parkinsonienne
# Design : Médical · Bleu/Blanc · Professionnel
# ============================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
# BASE DE DONNÉES
# ============================================
def init_db():
    conn = sqlite3.connect("neurogait.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient TEXT, age INTEGER, sexe TEXT,
        date TEXT, duree REAL,
        fog_episodes INTEGER, normal_windows INTEGER,
        score_risque INTEGER, cadence REAL,
        variabilite REAL, freeze_index REAL,
        utilisateur TEXT)""")
    conn.commit(); conn.close()

def save_session(patient, age, sexe, duree, n_fog, n_normal,
                 score, cadence, variabilite, fi, user):
    conn = sqlite3.connect("neurogait.db")
    conn.execute("""INSERT INTO sessions
        (patient,age,sexe,date,duree,fog_episodes,normal_windows,
         score_risque,cadence,variabilite,freeze_index,utilisateur)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (patient,age,sexe,datetime.now().strftime("%d/%m/%Y %H:%M"),
         round(duree,1),n_fog,n_normal,score,
         round(cadence,1),round(variabilite,3),round(fi,2),user))
    conn.commit(); conn.close()

def get_sessions():
    conn = sqlite3.connect("neurogait.db")
    df = pd.read_sql("SELECT * FROM sessions ORDER BY id DESC", conn)
    conn.close(); return df

init_db()

# ============================================
# CSS — DESIGN MÉDICAL BLEU/BLANC
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
    --white:   #FFFFFF;
    --grey:    #F7F9FC;
    --border:  #D6E4F0;
    --text:    #1A252F;
    --muted:   #5D7A8A;
    --danger:  #922B21;
    --success: #1A5632;
    --warn:    #784212;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--grey) !important;
    font-family: 'Source Sans 3', sans-serif !important;
    color: var(--text) !important;
}

/* ── SIDEBAR ─────────────────────────────── */
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
    color: white !important; border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem !important; font-weight: 500 !important;
    padding: 6px 0 !important;
}

/* ── LOGIN ───────────────────────────────── */
.login-wrap {
    max-width: 400px; margin: 60px auto 0;
    background: white; border-radius: 12px;
    padding: 44px 40px;
    box-shadow: 0 4px 32px rgba(13,33,55,0.12);
    border: 1px solid var(--border);
}
.login-logo {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.6rem; font-weight: 700;
    color: var(--navy); text-align: center;
    margin-bottom: 4px; letter-spacing: -0.3px;
}
.login-tagline {
    text-align: center; font-size: 0.83rem;
    color: var(--muted); margin-bottom: 32px; line-height: 1.5;
}
.login-field-label {
    font-size: 0.75rem; font-weight: 600;
    color: var(--muted); text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 5px;
    display: block;
}
.login-footer {
    text-align: center; font-size: 0.75rem;
    color: #B0BEC5; margin-top: 24px; line-height: 1.7;
}

/* ── PAGE HEADER ─────────────────────────── */
.page-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--blue) 65%, var(--mid) 100%);
    border-radius: 10px; padding: 30px 36px;
    margin-bottom: 24px;
    box-shadow: 0 2px 16px rgba(13,33,55,0.18);
    position: relative; overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute; right: -30px; top: -30px;
    width: 180px; height: 180px; border-radius: 50%;
    background: rgba(255,255,255,0.03);
    pointer-events: none;
}
.page-header-eyebrow {
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--light); margin-bottom: 8px;
}
.page-header-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.9rem; font-weight: 700;
    color: white; margin: 0 0 6px;
    letter-spacing: -0.4px; line-height: 1.2;
}
.page-header-sub {
    font-size: 0.88rem; color: rgba(255,255,255,0.6);
    font-weight: 300; letter-spacing: 0.3px;
}
.page-header-user {
    position: absolute; top: 20px; right: 24px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px; padding: 5px 14px;
    font-size: 0.8rem; color: rgba(255,255,255,0.8);
    font-weight: 500;
}

/* ── CARDS ───────────────────────────────── */
.card {
    background: white; border-radius: 10px;
    padding: 20px 22px;
    box-shadow: 0 1px 8px rgba(13,33,55,0.07);
    border: 1px solid var(--border);
}
.card-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 0.95rem; font-weight: 700;
    color: var(--navy); margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* ── KPI STRIP ───────────────────────────── */
.kpi {
    background: white; border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 6px rgba(13,33,55,0.06);
    border: 1px solid var(--border);
    border-top: 3px solid var(--mid);
}
.kpi.danger { border-top-color: var(--danger); }
.kpi.success { border-top-color: var(--success); }
.kpi.warn    { border-top-color: #E67E22; }
.kpi-lbl {
    font-size: 0.68rem; font-weight: 600;
    color: var(--muted); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 4px;
}
.kpi-val {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.75rem; font-weight: 700;
    color: var(--navy); line-height: 1.1;
}
.kpi-note { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* ── SCORE ───────────────────────────────── */
.score-ring {
    width: 110px; height: 110px; border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    margin: 0 auto 12px;
    font-family: 'Libre Baskerville', serif;
    font-weight: 700; color: white;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
}
.score-ring.low  { background: linear-gradient(135deg,#1A5632,#27AE60); }
.score-ring.med  { background: linear-gradient(135deg,#784212,#E67E22); }
.score-ring.high { background: linear-gradient(135deg,#641E16,#C0392B);
                   animation: pulse 2s infinite; }
@keyframes pulse {
    0%,100% { box-shadow:0 4px 16px rgba(192,57,43,.3); }
    50%      { box-shadow:0 4px 28px rgba(192,57,43,.65); }
}
.score-num { font-size: 1.8rem; line-height: 1; }
.score-denom { font-size: 0.7rem; opacity:.75; }
.score-label {
    font-family: 'Libre Baskerville', serif;
    font-size: 0.95rem; font-weight: 700;
    color: var(--navy); text-align: center;
}
.score-sublabel { font-size: 0.78rem; color: var(--muted); text-align: center; }

/* ── FEATURE ROW ─────────────────────────── */
.feat-row {
    display: flex; justify-content: space-between;
    align-items: center; padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.feat-row:last-child { border-bottom: none; }
.feat-name { color: var(--text); font-weight: 500; }
.feat-val  { font-weight: 600; font-family: 'Libre Baskerville', serif; }
.feat-ref  { font-size: 0.72rem; color: var(--muted); }
.tag {
    display: inline-block; font-size: 0.68rem; font-weight: 700;
    padding: 2px 8px; border-radius: 10px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.tag.ok  { background: #EAFAF1; color: #1A5632; }
.tag.nok { background: #FDEDEC; color: #922B21; }
.tag.med { background: #FEF9E7; color: #784212; }

/* ── INTERPRETATION ──────────────────────── */
.interp-box {
    background: var(--pale); border: 1px solid var(--border);
    border-left: 3px solid var(--mid);
    border-radius: 0 8px 8px 0;
    padding: 14px 18px; margin-bottom: 10px;
    font-size: 0.85rem; color: var(--text); line-height: 1.65;
}
.interp-box.critical {
    background: #FEF5F5; border-left-color: var(--danger);
}
.interp-title {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--blue); margin-bottom: 5px;
}
.interp-title.critical { color: var(--danger); }

/* ── DIAGNOSTIC BADGE ────────────────────── */
.dx-badge {
    border-radius: 8px; padding: 18px 20px;
    text-align: center; margin-bottom: 14px;
}
.dx-badge.fog    { background: linear-gradient(135deg,#922B21,#C0392B);
                   animation: pulse 2s infinite; }
.dx-badge.normal { background: linear-gradient(135deg,#1A5632,#27AE60); }
.dx-badge.wait   { background: linear-gradient(135deg,#1A252F,#2C3E50); }
.dx-icon  { font-size: 2rem; margin-bottom: 6px; }
.dx-title { font-family:'Libre Baskerville',serif; font-size:1.2rem;
            font-weight:700; color:white; }
.dx-sub   { font-size:.8rem; color:rgba(255,255,255,.7); margin-top:3px; }

/* ── PROBA BAR ───────────────────────────── */
.pbar-wrap { background:white; border-radius:8px; padding:14px 16px;
             border:1px solid var(--border); margin-bottom:12px; }
.pbar-lbl  { font-size:.68rem; font-weight:700; color:var(--muted);
             text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.pbar-bg   { background:#E8EDF2; border-radius:4px; height:8px; overflow:hidden; }
.pbar-fill { height:100%; border-radius:4px; transition:width .4s ease; }
.pbar-num  { font-family:'Libre Baskerville',serif; font-size:1.4rem;
             font-weight:700; margin-top:4px; }

/* ── ALERT BANNER ────────────────────────── */
.alert-crit {
    background: linear-gradient(90deg,#641E16,#922B21);
    border-radius: 8px; padding: 14px 18px; color: white;
    margin-bottom: 14px;
}
.alert-crit-title { font-weight:700; font-size:.9rem; }
.alert-crit-sub   { font-size:.78rem; opacity:.8; margin-top:2px; }

/* ── MISC ────────────────────────────────── */
.sec-label {
    font-size: .68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: var(--muted); margin-bottom: 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
}
.stButton > button {
    background: linear-gradient(135deg,var(--navy),var(--blue)) !important;
    color: white !important; border: none !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: .88rem !important;
    padding: 9px 20px !important; letter-spacing: .3px !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .82 !important; }
.stDownloadButton > button {
    background: white !important;
    color: var(--navy) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: .88rem !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none; }
hr { border-color: rgba(255,255,255,.08) !important; }
[data-testid="stFileUploader"] {
    background: white; border-radius: 8px;
    border: 1.5px dashed var(--border) !important;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
for k,v in [('authenticated',False),('username',''),
            ('running',False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================
# ML
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("models/model_fog_rf.pkl"), \
           joblib.load("models/scaler_fog.pkl")
try:
    model, scaler = load_model(); model_ok = True
except: model_ok = False

def bw_filter(sig, cutoff=3, fs=64, order=4):
    nyq = .5*fs; b,a = butter(order,cutoff/nyq,btype='low')
    return filtfilt(b,a,sig)

def extract_features(win, fs=64):
    feats = []
    for ax in range(win.shape[1]):
        s = win[:,ax]
        feats += [np.mean(s),np.std(s),np.max(s),np.min(s),
                  np.max(s)-np.min(s),np.sqrt(np.mean(s**2))]
        fv = np.abs(np.fft.rfft(s))
        fr = np.fft.rfftfreq(len(s),d=1/fs)
        feats.append(np.sum(fv))
        fz = fv[(fr>=3)&(fr<=8)]; lc = fv[(fr>=.5)&(fr<=3)]
        feats.append(np.sum(fz**2)/(np.sum(lc**2)+1e-6))
    return np.array(feats)

def predict_win(win,model,scaler):
    w=win.copy()
    for i in range(3): w[:,i]=bw_filter(w[:,i])
    f=extract_features(w).reshape(1,-1)
    fs=scaler.transform(f)
    pred=model.predict(fs)[0]; prob=model.predict_proba(fs)[0][1]
    return int(pred), round(float(prob)*100,1)

# ============================================
# SCORE DE RISQUE
# ============================================
def calc_score(cadence,variab,fi,prob_fog,n_fog,duree):
    s = 0
    if cadence<90: s+=20
    elif cadence<100: s+=10
    if variab>0.5: s+=25
    elif variab>0.2: s+=12
    if fi>3.0: s+=20
    elif fi>1.5: s+=10
    s += int(prob_fog*0.20)
    if duree>0:
        freq=(n_fog/duree)*60
        if freq>0.5: s+=15
        elif freq>0.2: s+=8
    return min(int(s),100)

def risk_level(score):
    if score<30:  return "Faible","low","#27AE60"
    if score<65:  return "Modéré","med","#E67E22"
    return "Élevé","high","#C0392B"

# ============================================
# INTERPRÉTATION SIGNAL
# ============================================
def interpret_signal(az_vals, cadence, variab, fi, fog_pred,
                     proba_moy, score, labels_real):
    """Génère une interprétation clinique détaillée du signal."""

    n_fog_real   = sum(labels_real)
    fog_pct      = (n_fog_real/max(len(labels_real),1))*100
    niveau,_,_   = risk_level(score)

    # Identification des zones anomales dans la courbe
    anomalies = []
    if cadence < 90:
        anomalies.append(
            "La cadence de marche est réduite à "
            f"<b>{cadence:.0f} pas/min</b> (norme ≈ 110 ppm). "
            "Sur la courbe, cela se traduit par un espacement irrégulier "
            "entre les pics d'impact du talon (heel strikes), "
            "visible entre les échantillons présentant des oscillations "
            "moins fréquentes."
        )
    if variab > 0.2:
        anomalies.append(
            f"La variabilité inter-foulée est élevée ({variab:.3f} s, "
            "norme &lt; 0.05 s). Cela se manifeste par des oscillations "
            "d'amplitude irrégulière dans la courbe — certains cycles sont "
            "plus longs, d'autres plus courts, sans régularité."
        )
    if fi > 1.5:
        anomalies.append(
            f"Le Freeze Index atteint <b>{fi:.2f}</b> (seuil pathologique = 1.5). "
            "Dans le spectre fréquentiel du signal, l'énergie dans la bande "
            "3–8 Hz (fréquence de freeze) dépasse celle de locomotion (0.5–3 Hz). "
            "Sur la courbe temporelle, cela correspond aux zones où l'amplitude "
            "du signal chute quasi à zéro, indiquant une immobilisation des membres."
        )
    if fog_pct > 10:
        anomalies.append(
            f"Les <b>zones rouges</b> sur la courbe représentent "
            f"{fog_pct:.1f}% de la session ({n_fog_real} fenêtres de 2 s). "
            "Ces segments correspondent aux épisodes de Freezing of Gait "
            "annotés cliniquement, où la marche est interrompue involontairement."
        )

    # Conclusion générale
    if score < 30:
        conclusion = (
            "Le profil de marche de ce patient est globalement dans les "
            "limites de la normale. Les paramètres cinématiques sont cohérents "
            "avec une démarche parkinsonienne stable et bien compensée. "
            "Un suivi de routine est suffisant."
        )
    elif score < 65:
        conclusion = (
            "Le profil de marche présente plusieurs indicateurs d'instabilité "
            "modérée. La variabilité temporelle et la réduction de cadence "
            "suggèrent une dégradation du contrôle locomoteur. "
            "Une consultation neurologique est recommandée pour ajuster "
            "le protocole thérapeutique."
        )
    else:
        conclusion = (
            "Le profil de marche est cliniquement préoccupant. La combinaison "
            "d'un Freeze Index élevé, d'une forte variabilité et d'épisodes "
            "FoG fréquents indique un risque élevé de chutes. "
            "Une consultation neurologique urgente est fortement recommandée."
        )

    return anomalies, conclusion

# ============================================
# RAPPORT PDF
# ============================================
def gen_pdf(patient,age,sexe,date,duree,n_fog,n_normal,
            score,cadence,variab,fi,user):
    pdf = FPDF(); pdf.add_page()
    pdf.set_auto_page_break(auto=True,margin=15)
    niveau,_,_ = risk_level(score)

    pdf.set_fill_color(13,33,55); pdf.rect(0,0,210,38,'F')
    pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",18)
    pdf.set_xy(15,10); pdf.cell(0,8,"NeuroGait — Rapport d'Analyse de Marche",ln=True)
    pdf.set_font("Helvetica","",9); pdf.set_xy(15,22)
    pdf.cell(0,6,"Système de Détection du Freezing of Gait — PFE Ingénierie Biomédicale 2026")

    pdf.set_xy(15,46); pdf.set_text_color(13,33,55)
    pdf.set_font("Helvetica","B",12); pdf.cell(0,7,"Informations Patient",ln=True)
    pdf.set_draw_color(46,134,193); pdf.set_line_width(.4)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)

    infos=[("Patient",patient),("Âge",f"{age} ans"),("Sexe",sexe),
           ("Date",date),("Utilisateur",user)]
    for lbl,val in infos:
        pdf.set_font("Helvetica","B",9); pdf.cell(55,6,f"{lbl} :",ln=False)
        pdf.set_font("Helvetica","",9); pdf.cell(0,6,val,ln=True)

    pdf.ln(5); pdf.set_font("Helvetica","B",12)
    pdf.cell(0,7,"Score de Risque Global",ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)

    pdf.set_fill_color(230,235,240); pdf.rect(15,pdf.get_y(),180,10,'F')
    if score<30: pdf.set_fill_color(26,86,50)
    elif score<65: pdf.set_fill_color(120,66,18)
    else: pdf.set_fill_color(100,30,22)
    pdf.rect(15,pdf.get_y(),int(180*score/100),10,'F')
    pdf.set_text_color(255,255,255); pdf.set_font("Helvetica","B",9)
    pdf.set_xy(15,pdf.get_y()+1)
    pdf.cell(180,8,f"  {score}/100 — Risque {niveau}",ln=True)
    pdf.ln(2); pdf.set_text_color(13,33,55)
    pdf.set_font("Helvetica","",9)
    if score<30:
        pdf.multi_cell(0,5,"Paramètres de marche dans les limites normales. Surveillance de routine recommandée.")
    elif score<65:
        pdf.multi_cell(0,5,"Anomalies modérées détectées. Consultation neurologique conseillée.")
    else:
        pdf.multi_cell(0,5,"RISQUE ÉLEVÉ. Épisodes FoG sévères détectés. Consultation urgente recommandée.")

    pdf.ln(5); pdf.set_font("Helvetica","B",12)
    pdf.cell(0,7,"Biomarqueurs de la Marche",ln=True)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)

    hdrs=["Biomarqueur","Valeur","Norme","Statut"]; cw=[70,35,45,35]
    pdf.set_fill_color(13,33,55); pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",9)
    for i,h in enumerate(hdrs): pdf.cell(cw[i],7,h,border=1,fill=True)
    pdf.ln()

    rows=[("Durée session",f"{duree:.0f} s","—","—"),
          ("Cadence",f"{cadence:.1f} ppm","~110 ppm","Anormal" if cadence<90 else "Normal"),
          ("Variabilité",f"{variab:.3f} s","< 0.05 s","Anormal" if variab>0.2 else "Normal"),
          ("Freeze Index",f"{fi:.2f}","< 1.5","Anormal" if fi>1.5 else "Normal"),
          ("Épisodes FoG",str(n_fog),"0","Présent" if n_fog>0 else "Absent"),
          ("Fenêtres normales",str(n_normal),"—","—")]
    pdf.set_font("Helvetica","",9)
    for i,row in enumerate(rows):
        pdf.set_fill_color(247,249,252) if i%2==0 else pdf.set_fill_color(255,255,255)
        pdf.set_text_color(13,33,55)
        for j,c in enumerate(row):
            if j==3 and c in ["Anormal","Présent"]: pdf.set_text_color(146,43,33)
            elif j==3: pdf.set_text_color(26,86,50)
            pdf.cell(cw[j],6,c,border=1,fill=True)
            pdf.set_text_color(13,33,55)
        pdf.ln()

    pdf.set_y(-28); pdf.set_draw_color(46,134,193)
    pdf.line(15,pdf.get_y(),195,pdf.get_y()); pdf.ln(3)
    pdf.set_font("Helvetica","",8); pdf.set_text_color(150,160,170)
    pdf.cell(0,5,f"Généré le {date} — NeuroGait · PFE Ingénierie Biomédicale 2026",ln=True,align='C')
    pdf.cell(0,5,"Document généré automatiquement. Ne remplace pas un diagnostic médical professionnel.",ln=True,align='C')
    return bytes(pdf.output())

# ============================================
# GMAIL
# ============================================
def send_gmail(dest,patient,n_fog,score,gmail_user,gmail_pass,pdf_b=None):
    try:
        msg=MIMEMultipart(); msg['From']=gmail_user
        msg['To']=dest; niveau,_,_=risk_level(score)
        msg['Subject']=f"Alerte NeuroGait — FoG détecté chez {patient}"
        body=f"""Bonjour,

Le système NeuroGait a identifié une activité de marche nécessitant votre attention.

Patient         : {patient}
Date            : {datetime.now().strftime("%d/%m/%Y à %H:%M")}
Épisodes FoG    : {n_fog}
Score de risque : {score}/100 — Risque {niveau}

Une consultation est recommandée. Le rapport détaillé est joint en pièce jointe.

Cordialement,
Système NeuroGait — PFE Ingénierie Biomédicale"""
        msg.attach(MIMEText(body,'plain','utf-8'))
        if pdf_b:
            p=MIMEBase('application','octet-stream'); p.set_payload(pdf_b)
            encoders.encode_base64(p)
            p.add_header('Content-Disposition',f'attachment; filename="rapport_{patient}.pdf"')
            msg.attach(p)
        srv=smtplib.SMTP_SSL('smtp.gmail.com',465)
        srv.login(gmail_user,gmail_pass); srv.sendmail(gmail_user,dest,msg.as_string())
        srv.quit(); return True,"Email envoyé avec succès."
    except Exception as e: return False,str(e)

# ============================================
# PAGE LOGIN
# ============================================
if not st.session_state['authenticated']:
    st.markdown("""<style>
    [data-testid="stSidebar"]{display:none!important;}
    .block-container{padding:0!important;max-width:100%!important;}
    </style>""",unsafe_allow_html=True)

    _,c,_ = st.columns([1,1.1,1])
    with c:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-bottom:4px;
                    font-size:2.4rem;'>🔬</div>
        <div style='font-family:Libre Baskerville,serif;font-size:1.9rem;
                    font-weight:700;color:#0D2137;text-align:center;
                    margin-bottom:4px;'>NeuroGait</div>
        <div style='font-size:.83rem;color:#5D7A8A;text-align:center;
                    margin-bottom:32px;line-height:1.5;'>
            Système d'Analyse de la Marche Parkinsonienne<br>
            PFE — Ingénierie Biomédicale
        </div>
        """,unsafe_allow_html=True)

        username = st.text_input("Identifiant", placeholder="ex : medecin")
        password = st.text_input("Mot de passe", type="password",
                                 placeholder="••••••••")
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("Connexion", use_container_width=True):
            if check_password(username,password):
                st.session_state['authenticated']=True
                st.session_state['username']=username
                st.rerun()
            else:
                st.error("Identifiant ou mot de passe incorrect.")

        st.markdown("""
        <div style='text-align:center;margin-top:20px;font-size:.75rem;
                    color:#B0BEC5;line-height:1.7;'>
            Accès réservé au personnel autorisé<br>
            © 2026 NeuroGait
        </div>""",unsafe_allow_html=True)

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
    <div style='padding:24px 0 16px;text-align:center;'>
        <div style='font-size:2.2rem;'>🔬</div>
        <div style='font-family:Libre Baskerville,serif;font-size:1.15rem;
                    font-weight:700;color:white;margin-top:6px;'>NeuroGait</div>
        <div style='font-size:.65rem;color:rgba(255,255,255,.4);
                    letter-spacing:2px;text-transform:uppercase;'>
            Analyse de Marche
        </div>
    </div>
    <hr>
    <div style='background:rgba(255,255,255,.07);border-radius:7px;
                padding:9px 13px;margin-bottom:16px;'>
        <div style='font-size:.65rem;color:rgba(255,255,255,.4);
                    text-transform:uppercase;letter-spacing:1px;'>Session</div>
        <div style='font-size:.9rem;font-weight:600;color:white;margin-top:2px;'>
            {st.session_state['username']}
        </div>
    </div>
    <div class='sec-label' style='color:rgba(255,255,255,.35);border-color:rgba(255,255,255,.08);'>
        Navigation
    </div>
    """,unsafe_allow_html=True)

    page = st.radio("",["Analyse Fichier","Démo Temps Réel",
                        "Historique","Configuration"],
                    label_visibility="collapsed")

    st.markdown("""<hr><div class='sec-label'
    style='color:rgba(255,255,255,.35);border-color:rgba(255,255,255,.08);'>
    Dossier Patient</div>""",unsafe_allow_html=True)

    nom = st.text_input("Nom", "Patient 001", label_visibility="collapsed")
    c1,c2 = st.columns(2)
    with c1: age  = st.number_input("Âge",0,120,65,label_visibility="collapsed")
    with c2: sexe = st.selectbox("Sexe",["H","F"],label_visibility="collapsed")

    st.markdown("<hr>",unsafe_allow_html=True)
    if model_ok:
        st.markdown("<div style='background:rgba(26,86,50,.25);border:1px solid rgba(39,174,96,.3);border-radius:6px;padding:8px 12px;font-size:.78rem;color:#7DCEA0;font-weight:600;'>Modèle ML chargé</div>",unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:rgba(146,43,33,.2);border:1px solid rgba(192,57,43,.3);border-radius:6px;padding:8px 12px;font-size:.78rem;color:#F1948A;font-weight:600;'>Modèle non trouvé</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("Déconnexion",use_container_width=True):
        st.session_state['authenticated']=False
        st.session_state['username']=''
        st.rerun()

# ============================================
# HEADER
# ============================================
st.markdown(f"""
<div class="page-header">
    <div class="page-header-user">{st.session_state['username']}</div>
    <div class="page-header-eyebrow">Système de Diagnostic Médical</div>
    <div class="page-header-title">Analyse de la Marche Parkinsonienne</div>
    <div class="page-header-sub">
        Détection FoG · Score de Risque · Interprétation Clinique · Rapport PDF
    </div>
</div>
""",unsafe_allow_html=True)

# ============================================
# PAGE : ANALYSE FICHIER
# ============================================
if page == "Analyse Fichier":

    st.markdown('<div class="sec-label">Chargement des données</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Fichier .txt (Daphnet) ou .csv",
        type=["txt","csv"], label_visibility="collapsed"
    )

    if not uploaded:
        st.markdown("""
        <div style='background:white;border-radius:10px;padding:56px;
                    text-align:center;border:1px solid #D6E4F0;'>
            <div style='font-size:2.8rem;margin-bottom:14px;'>📂</div>
            <div style='font-family:Libre Baskerville,serif;font-size:1.25rem;
                        font-weight:700;color:#0D2137;margin-bottom:8px;'>
                Aucun fichier chargé
            </div>
            <div style='color:#5D7A8A;font-size:.88rem;max-width:340px;
                        margin:0 auto;line-height:1.6;'>
                Glissez un fichier <b>.txt</b> du dataset Daphnet FoG
                ou un fichier CSV pour démarrer l'analyse.
            </div>
        </div>""",unsafe_allow_html=True)
        st.stop()

    # ---- Chargement ----
    col_names=['time','ankle_x','ankle_y','ankle_z',
               'thigh_x','thigh_y','thigh_z',
               'trunk_x','trunk_y','trunk_z','label']
    try:
        df = pd.read_csv(uploaded,sep=" ",header=None,names=col_names)
        df = df[df['label']!=0].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erreur de lecture : {e}"); st.stop()

    # ---- Pipeline ML ----
    WS,OL = 128,64
    windows_t,labels_pred,labels_real,probas_list = [],[],[],[]

    for start in range(0,len(df)-WS,OL):
        end  = start+WS
        win  = df[['ankle_x','ankle_y','ankle_z']].iloc[start:end].values
        lbl  = df['label'].iloc[start:end].mode()[0]
        if model_ok:
            pred,prob = predict_win(win,model,scaler)
            labels_pred.append(pred); probas_list.append(prob)
        labels_real.append(1 if lbl==2 else 0)
        windows_t.append(start/64)

    fog_pred  = sum(labels_pred) if model_ok else 0
    fog_real  = sum(labels_real)
    duree     = len(df)/64
    az        = df['ankle_z'].values

    # Biomarqueurs
    try:
        peaks,_ = find_peaks(bw_filter(az),distance=32)
        if len(peaks)>2:
            diffs    = np.diff(peaks)/64
            cadence  = float(60/np.mean(diffs))
            variab   = float(np.std(diffs))
        else:
            cadence,variab = 100.0,0.1
    except: cadence,variab = 100.0,0.1
    cadence = min(max(cadence,50),150); variab=min(variab,2.0)

    fv=np.abs(np.fft.rfft(az[:512]))
    fr=np.fft.rfftfreq(512,d=1/64)
    fz=fv[(fr>=3)&(fr<=8)]; lc=fv[(fr>=.5)&(fr<=3)]
    fi_moy = float(np.sum(fz**2)/(np.sum(lc**2)+1e-6))
    fi_moy = min(fi_moy,10.0)

    proba_moy = float(np.mean(probas_list)) if probas_list else 0.0
    score     = calc_score(cadence,variab,fi_moy,proba_moy/100,fog_pred,duree)
    niveau,sc_cls,sc_col = risk_level(score)
    anomalies,conclusion = interpret_signal(az,cadence,variab,fi_moy,
                                            fog_pred,proba_moy,score,labels_real)

    # ============================================================
    # LAYOUT PRINCIPAL
    # ============================================================
    # ---- KPI strip ----
    k1,k2,k3,k4,k5 = st.columns(5)
    cards = [
        (k1,"","Durée session",f"{duree:.0f} s",f"{len(df):,} points · 64 Hz"),
        (k2,"success","Marche normale",
         str(len(labels_pred)-fog_pred if labels_pred else 0),
         "fenêtres classifiées"),
        (k3,"danger","Épisodes FoG",str(fog_pred),"détectés par IA"),
        (k4,"warn" if cadence<100 else "success","Cadence",
         f"{cadence:.0f}","pas / minute"),
        (k5,"danger" if score>=65 else ("warn" if score>=30 else "success"),
         "Score de risque",str(score),f"Risque {niveau}"),
    ]
    for col,cls,lbl,val,note in cards:
        with col:
            st.markdown(f'<div class="kpi {cls}"><div class="kpi-lbl">{lbl}</div>'
                        f'<div class="kpi-val">{val}</div>'
                        f'<div class="kpi-note">{note}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    # ---- Signal + Score + Features ----
    col_sig, col_right = st.columns([3,1])

    with col_sig:
        # Graphique signal annoté
        st.markdown('<div class="sec-label">Signal accéléromètre — cheville (axe Z)</div>',
                    unsafe_allow_html=True)

        fig = go.Figure()

        # Signal filtré
        az_f = bw_filter(az)
        fig.add_trace(go.Scatter(
            y=az_f[:3000], mode='lines', name='Signal filtré',
            line=dict(color='#1A5276',width=1.4),
            hovertemplate='Valeur : %{y:.1f} mg<extra></extra>'
        ))

        # Signal brut en transparence
        fig.add_trace(go.Scatter(
            y=az[:3000], mode='lines', name='Signal brut',
            line=dict(color='#AED6F1',width=.7,dash='dot'),
            opacity=.5,
            hovertemplate='Brut : %{y:.1f} mg<extra></extra>'
        ))

        # Zones FoG annotées
        in_fog=False; sf=0; n_zone=0
        for i in range(min(3000,len(df))):
            if df['label'].iloc[i]==2 and not in_fog:
                sf=i; in_fog=True
            elif df['label'].iloc[i]!=2 and in_fog:
                fig.add_vrect(x0=sf,x1=i,fillcolor="#C0392B",
                              opacity=.1,line_width=0)
                # Annotation sur la première zone
                if n_zone==0:
                    fig.add_annotation(
                        x=(sf+i)//2, y=az_f[:3000].max()*1.05,
                        text="Épisode FoG",
                        showarrow=True, arrowhead=2,
                        arrowcolor="#922B21", arrowsize=1.2,
                        font=dict(size=10,color="#922B21"),
                        ax=0, ay=-30
                    )
                n_zone+=1; in_fog=False

        # Annotation marche normale
        fig.add_annotation(
            x=30, y=az_f[:3000].min()*1.1,
            text="Marche régulière → pics périodiques",
            showarrow=False,
            font=dict(size=9,color="#1A5276"),
            bgcolor="rgba(235,245,251,.85)",
            bordercolor="#AED6F1", borderwidth=1,
            borderpad=4
        )

        fig.update_layout(
            height=300, margin=dict(l=10,r=10,t=10,b=10),
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(color='#1A252F',family='Source Sans 3'),
            xaxis=dict(title="Échantillons",showgrid=True,
                       gridcolor='#F0F4F8',linecolor='#D6E4F0',
                       tickfont=dict(size=10)),
            yaxis=dict(title="mg",showgrid=True,
                       gridcolor='#F0F4F8',linecolor='#D6E4F0',
                       tickfont=dict(size=10)),
            legend=dict(orientation='h',yanchor='bottom',y=1.02,
                        font=dict(size=10),bgcolor='rgba(0,0,0,0)'),
            hovermode='x unified'
        )
        st.plotly_chart(fig,use_container_width=True,
                        config={'displayModeBar':False})

        # Probabilité FoG dans le temps
        if model_ok and probas_list:
            st.markdown('<div class="sec-label">Probabilité FoG au fil du temps</div>',
                        unsafe_allow_html=True)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=windows_t, y=probas_list, mode='lines',
                fill='tozeroy',
                line=dict(color='#1A5276',width=1.5),
                fillcolor='rgba(26,82,118,.08)',
                name='P(FoG)',
                hovertemplate='t=%{x:.0f}s · P(FoG)=%{y:.1f}%<extra></extra>'
            ))
            # Zone critique
            fig2.add_hrect(y0=60,y1=105,
                           fillcolor="rgba(192,57,43,.06)",
                           line_width=0)
            fig2.add_hline(y=60,line_dash="dash",
                           line_color="#C0392B",line_width=1,
                           annotation_text="Seuil 60 %",
                           annotation_font=dict(size=9,color="#922B21"))
            fig2.update_layout(
                height=180,margin=dict(l=10,r=10,t=10,b=10),
                plot_bgcolor='white',paper_bgcolor='white',
                font=dict(color='#1A252F',family='Source Sans 3'),
                xaxis=dict(title="Temps (s)",showgrid=True,
                           gridcolor='#F0F4F8',tickfont=dict(size=10)),
                yaxis=dict(title="%",range=[0,105],showgrid=True,
                           gridcolor='#F0F4F8',tickfont=dict(size=10)),
                showlegend=False,hovermode='x unified')
            st.plotly_chart(fig2,use_container_width=True,
                            config={'displayModeBar':False})

    with col_right:
        # Score de risque
        st.markdown(f"""
        <div class="card" style="text-align:center;margin-bottom:14px;">
            <div class="card-title">Score de risque</div>
            <div class="score-ring {sc_cls}">
                <div class="score-num">{score}</div>
                <div class="score-denom">/100</div>
            </div>
            <div class="score-label">Risque {niveau}</div>
            <div class="score-sublabel">{datetime.now().strftime("%d/%m/%Y")}</div>
        </div>
        """,unsafe_allow_html=True)

        # Détail features
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown('<div class="card-title">Biomarqueurs détaillés</div>',
                    unsafe_allow_html=True)

        features_detail = [
            ("Cadence",f"{cadence:.1f} ppm","~110 ppm",
             "ok" if cadence>=100 else ("med" if cadence>=85 else "nok")),
            ("Variabilité",f"{variab:.3f} s","< 0.05 s",
             "ok" if variab<0.05 else ("med" if variab<0.2 else "nok")),
            ("Freeze Index",f"{fi_moy:.2f}","< 1.5",
             "ok" if fi_moy<1.5 else ("med" if fi_moy<3.0 else "nok")),
            ("P(FoG) moy.",f"{proba_moy:.1f}%","< 30%",
             "ok" if proba_moy<30 else ("med" if proba_moy<60 else "nok")),
            ("FoG détectés",str(fog_pred),"0",
             "ok" if fog_pred==0 else ("med" if fog_pred<=3 else "nok")),
        ]
        tag_labels = {"ok":"Normal","med":"Modéré","nok":"Anormal"}

        for fname,fval,fref,ftag in features_detail:
            tlabel = tag_labels[ftag]
            st.markdown(f"""
            <div class="feat-row">
                <div>
                    <div class="feat-name">{fname}</div>
                    <div class="feat-ref">Réf : {fref}</div>
                </div>
                <div style="text-align:right;">
                    <div class="feat-val">{fval}</div>
                    <span class="tag {ftag}">{tlabel}</span>
                </div>
            </div>""",unsafe_allow_html=True)

        st.markdown('</div>',unsafe_allow_html=True)

    # ============================================================
    # INTERPRÉTATION CLINIQUE
    # ============================================================
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Interprétation clinique du signal</div>',
                unsafe_allow_html=True)

    if anomalies:
        for i, anom in enumerate(anomalies):
            critical = "FoG" in anom or "Freeze" in anom or "rouge" in anom
            css = "critical" if critical else ""
            st.markdown(f"""
            <div class="interp-box {css}">
                <div class="interp-title {css}">
                    Observation {i+1}
                </div>
                {anom}
            </div>""",unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="interp-box">
            <div class="interp-title">Observations</div>
            Le signal présente un profil rythmique régulier, sans anomalie
            notable. Les pics d'accélération sont espacés uniformément,
            traduisant un cycle de marche stable.
        </div>""",unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box {'critical' if score>=65 else ''}">
        <div class="interp-title {'critical' if score>=65 else ''}">
            Conclusion générale
        </div>
        {conclusion}
    </div>""",unsafe_allow_html=True)

    # ============================================================
    # ACTIONS
    # ============================================================
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Actions</div>',
                unsafe_allow_html=True)

    pdf_bytes = gen_pdf(nom,age,sexe,
                        datetime.now().strftime("%d/%m/%Y %H:%M"),
                        duree,fog_pred,
                        len(labels_pred)-fog_pred if labels_pred else 0,
                        score,cadence,variab,fi_moy,
                        st.session_state['username'])

    a1,a2,a3 = st.columns(3)
    with a1:
        st.download_button(
            "Télécharger le rapport PDF",
            data=pdf_bytes,
            file_name=f"rapport_{nom}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with a2:
        if st.button("Sauvegarder dans l'historique",
                     use_container_width=True):
            save_session(nom,age,sexe,duree,fog_pred,
                         len(labels_pred)-fog_pred if labels_pred else 0,
                         score,cadence,variab,fi_moy,
                         st.session_state['username'])
            st.success("Session enregistrée.")

    if score>=65 and fog_pred>0:
        st.markdown(f"""
        <div class="alert-crit">
            <div class="alert-crit-title">Risque élevé détecté</div>
            <div class="alert-crit-sub">
                {fog_pred} épisodes FoG · Score {score}/100 ·
                Consultez l'onglet Configuration pour envoyer une alerte email.
            </div>
        </div>""",unsafe_allow_html=True)

# ============================================
# PAGE : DÉMO TEMPS RÉEL
# ============================================
elif page == "Démo Temps Réel":

    col_c, col_d = st.columns([3,1])
    with col_d:
        st.markdown('<div class="sec-label">Diagnostic</div>',
                    unsafe_allow_html=True)
        dx=st.empty(); pb=st.empty(); sb=st.empty()
        dx.markdown('<div class="dx-badge wait"><div class="dx-icon">◌</div>'
                    '<div class="dx-title">En attente</div>'
                    '<div class="dx-sub">Démarrez la session</div></div>',
                    unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="sec-label">Signal IMU — temps réel</div>',
                    unsafe_allow_html=True)
        ch=st.empty()

    c1,c2,_ = st.columns([1,1,5])
    with c1: s1=st.button("Démarrer",type="primary",use_container_width=True)
    with c2: s2=st.button("Arrêter",use_container_width=True)
    if s1: st.session_state['running']=True
    if s2: st.session_state['running']=False

    if st.session_state.get('running',False):
        buf=collections.deque(maxlen=128)
        hx,hy,hz=[],[],[]
        nf=nn=t=0; lp=0; lpr=0.0

        while st.session_state.get('running',False):
            is_fog=(t%320>230)
            if is_fog:
                ax=np.random.normal(300,90)
                ay=np.random.normal(950,130)
                az_=np.random.normal(100,210)+160*np.sin(2*np.pi*5*t/64)
            else:
                ax=np.random.normal(50,20)
                ay=np.random.normal(980,30)
                az_=np.random.normal(-80,40)+65*np.sin(2*np.pi*1.8*t/64)

            buf.append([ax,ay,az_])
            hx.append(ax); hy.append(ay); hz.append(az_)
            if len(hx)>320: hx=hx[-320:]; hy=hy[-320:]; hz=hz[-320:]

            if len(buf)==128 and model_ok:
                lp,lpr=predict_win(np.array(buf),model,scaler)
                if lp==1: nf+=1
                else: nn+=1

            if lp==1:
                dx.markdown('<div class="dx-badge fog"><div class="dx-icon">●</div>'
                            '<div class="dx-title">FoG détecté</div>'
                            '<div class="dx-sub">Enrayement cinétique</div></div>',
                            unsafe_allow_html=True)
            else:
                dx.markdown('<div class="dx-badge normal"><div class="dx-icon">●</div>'
                            '<div class="dx-title">Marche normale</div>'
                            '<div class="dx-sub">Aucune anomalie</div></div>',
                            unsafe_allow_html=True)

            bc="#C0392B" if lpr>60 else "#1A5276"
            pb.markdown(f"""
            <div class="pbar-wrap">
                <div class="pbar-lbl">Probabilité FoG</div>
                <div class="pbar-bg">
                    <div class="pbar-fill" style="width:{lpr}%;background:{bc};"></div>
                </div>
                <div class="pbar-num" style="color:{bc};">{lpr}%</div>
            </div>""",unsafe_allow_html=True)

            sb.markdown(f"""
            <div class="card">
                <div class="feat-row">
                    <span class="feat-name">Normale</span>
                    <span class="feat-val" style="color:#1A5632;">{nn}</span>
                </div>
                <div class="feat-row">
                    <span class="feat-name">FoG</span>
                    <span class="feat-val" style="color:#922B21;">{nf}</span>
                </div>
            </div>""",unsafe_allow_html=True)

            lc='#C0392B' if is_fog else '#1A5276'
            fig=go.Figure()
            fig.add_trace(go.Scatter(y=hx,mode='lines',name='X',
                line=dict(color='#AED6F1',width=1)))
            fig.add_trace(go.Scatter(y=hy,mode='lines',name='Y',
                line=dict(color='#A9DFBF',width=1)))
            fig.add_trace(go.Scatter(y=hz,mode='lines',name='Z',
                line=dict(color=lc,width=1.8)))
            fig.update_layout(height=320,
                margin=dict(l=10,r=10,t=10,b=10),
                plot_bgcolor='white',paper_bgcolor='white',
                font=dict(color='#1A252F',family='Source Sans 3'),
                xaxis=dict(showgrid=True,gridcolor='#F0F4F8'),
                yaxis=dict(title='mg',showgrid=True,gridcolor='#F0F4F8'),
                legend=dict(orientation='h',yanchor='bottom',y=1.02,
                            font=dict(size=10),bgcolor='rgba(0,0,0,0)'))
            ch.plotly_chart(fig,use_container_width=True,
                            config={'displayModeBar':False})
            t+=1; time.sleep(0.04)

# ============================================
# PAGE : HISTORIQUE
# ============================================
elif page == "Historique":

    st.markdown('<div class="sec-label">Historique des sessions</div>',
                unsafe_allow_html=True)
    try:
        df_h=get_sessions()
        if len(df_h)==0:
            st.info("Aucune session enregistrée. "
                    "Analysez un fichier et cliquez sur 'Sauvegarder'.")
        else:
            h1,h2,h3,h4=st.columns(4)
            for col,cls,lbl,val,note in [
                (h1,"","Sessions",len(df_h),"enregistrées"),
                (h2,"danger","FoG total",int(df_h['fog_episodes'].sum()),"épisodes"),
                (h3,"warn","Score moyen",f"{df_h['score_risque'].mean():.0f}","/100"),
                (h4,"success","Patients",df_h['patient'].nunique(),"uniques")]:
                with col:
                    st.markdown(f'<div class="kpi {cls}"><div class="kpi-lbl">'
                                f'{lbl}</div><div class="kpi-val">{val}</div>'
                                f'<div class="kpi-note">{note}</div></div>',
                                unsafe_allow_html=True)

            st.markdown("<br>",unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Évolution du score de risque</div>',
                        unsafe_allow_html=True)

            fig_h=go.Figure()
            for pt in df_h['patient'].unique():
                dp=df_h[df_h['patient']==pt]
                fig_h.add_trace(go.Scatter(
                    x=dp['date'],y=dp['score_risque'],
                    mode='lines+markers',name=pt,line=dict(width=2)))
            fig_h.add_hline(y=65,line_dash="dash",
                            line_color="#C0392B",line_width=1,
                            annotation_text="Seuil risque élevé",
                            annotation_font=dict(size=9,color="#922B21"))
            fig_h.update_layout(height=260,
                plot_bgcolor='white',paper_bgcolor='white',
                font=dict(color='#1A252F',family='Source Sans 3'),
                xaxis=dict(title="Date",showgrid=True,gridcolor='#F0F4F8'),
                yaxis=dict(title="Score",range=[0,105],showgrid=True,
                           gridcolor='#F0F4F8'),
                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_h,use_container_width=True,
                            config={'displayModeBar':False})

            st.markdown('<div class="sec-label">Détail</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                df_h[['patient','age','date','duree',
                      'fog_episodes','score_risque','utilisateur']]
                .rename(columns={
                    'patient':'Patient','age':'Âge','date':'Date',
                    'duree':'Durée (s)','fog_episodes':'FoG',
                    'score_risque':'Score','utilisateur':'Utilisateur'}),
                use_container_width=True,hide_index=True)
    except Exception as e:
        st.error(f"Erreur : {e}")

# ============================================
# PAGE : CONFIGURATION
# ============================================
elif page == "Configuration":

    st.markdown('<div class="sec-label">Configuration des alertes email (Gmail)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="interp-box">
        <div class="interp-title">Configuration requise</div>
        Activez la <b>validation en 2 étapes</b> sur votre compte Gmail, puis
        générez un <b>mot de passe d'application</b> dans
        Paramètres → Sécurité → Mots de passe des applications.
        Utilisez ce mot de passe de 16 caractères ci-dessous,
        et non votre mot de passe habituel.
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Expéditeur**")
        gu=st.text_input("Adresse Gmail",placeholder="votre.email@gmail.com")
        gp=st.text_input("Mot de passe app",type="password",
                         placeholder="xxxx xxxx xxxx xxxx")
    with c2:
        st.markdown("**Destinataire**")
        de=st.text_input("Email médecin",placeholder="medecin@hopital.ma")
        sf=st.slider("Alerte si FoG ≥",1,10,3)
        ss=st.slider("Alerte si Score ≥",30,100,65)

    st.markdown("<br>",unsafe_allow_html=True)

    if st.button("Envoyer un email de test"):
        if not gu or not gp or not de:
            st.error("Veuillez remplir tous les champs.")
        else:
            pb=gen_pdf("Patient Test",65,"H",
                       datetime.now().strftime("%d/%m/%Y %H:%M"),
                       120,5,45,72,88.5,.45,2.8,
                       st.session_state['username'])
            ok,msg=send_gmail(de,"Patient Test",5,72,gu,gp,pb)
            if ok: st.success(msg)
            else: st.error(msg)

    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown("**Envoi manuel**")
    m1,m2=st.columns(2)
    with m1:
        pa=st.text_input("Patient",nom)
        fa=st.number_input("Épisodes FoG",0,100,5)
    with m2:
        sa=st.number_input("Score de risque",0,100,72)

    if st.button("Envoyer l'alerte",type="primary"):
        if not gu or not gp or not de:
            st.error("Configurez d'abord Gmail ci-dessus.")
        else:
            pb=gen_pdf(pa,age,sexe,
                       datetime.now().strftime("%d/%m/%Y %H:%M"),
                       0,fa,0,sa,100.,.5,2.,
                       st.session_state['username'])
            ok,msg=send_gmail(de,pa,fa,sa,gu,gp,pb)
            if ok: st.success(msg)
            else: st.error(msg)
