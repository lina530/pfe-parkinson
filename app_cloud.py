# ============================================
# app_cloud.py — NeuroGait
# Système de Détection Parkinson
# Fonctionnalités :
#   1. Score de Risque Patient
#   2. Rapport PDF automatique
#   3. Alerte Gmail en cas de FoG sévère
# ============================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
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
import io

st.set_page_config(
    page_title="NeuroGait — Détection Parkinson",
    page_icon="🧠",
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

def check_password(username, password):
    return USERS.get(username) == hashlib.sha256(password.encode()).hexdigest()

# ============================================
# BASE DE DONNÉES SQLite
# ============================================
def init_db():
    conn = sqlite3.connect("neurogait.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient TEXT, age INTEGER, sexe TEXT,
            date TEXT, duree REAL,
            fog_episodes INTEGER, normal_windows INTEGER,
            score_risque INTEGER,
            cadence REAL, variabilite REAL, freeze_index REAL,
            utilisateur TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_session(patient, age, sexe, duree, n_fog, n_normal,
                 score, cadence, variabilite, fi, user):
    conn = sqlite3.connect("neurogait.db")
    conn.execute("""
        INSERT INTO sessions
        (patient, age, sexe, date, duree, fog_episodes,
         normal_windows, score_risque, cadence, variabilite,
         freeze_index, utilisateur)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (patient, age, sexe,
          datetime.now().strftime("%d/%m/%Y %H:%M"),
          round(duree,1), n_fog, n_normal, score,
          round(cadence,1), round(variabilite,3),
          round(fi,2), user))
    conn.commit()
    conn.close()

def get_sessions():
    conn = sqlite3.connect("neurogait.db")
    df = pd.read_sql("SELECT * FROM sessions ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# ============================================
# CSS DESIGN
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #F4F6F9 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background: #F4F6F9 !important; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A2342 0%, #1B4F72 100%) !important;
}
[data-testid="stSidebar"] * { color: #E8EDF2 !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important; border-radius: 8px !important;
}

/* HERO */
.hero-container {
    background: linear-gradient(135deg, #0A2342 0%, #1B4F72 55%, #148F77 100%);
    border-radius: 16px; padding: 36px 40px; margin-bottom: 28px;
    position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(10,35,66,0.25);
}
.hero-container::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background:rgba(255,255,255,0.04);
}
.hero-badge {
    display:inline-block; background:rgba(20,143,119,0.3);
    border:1px solid rgba(20,143,119,0.6); color:#7DCEA0;
    font-size:0.75rem; font-weight:600; letter-spacing:1.5px;
    text-transform:uppercase; padding:4px 12px;
    border-radius:20px; margin-bottom:14px;
}
.hero-title {
    font-family:'Playfair Display',serif; font-size:2.2rem;
    font-weight:700; color:#FFFFFF; margin:0 0 6px 0;
}
.hero-subtitle {
    font-size:0.95rem; color:rgba(255,255,255,0.65);
    font-weight:300; letter-spacing:0.5px; margin:0;
}
.hero-user {
    position:absolute; top:24px; right:28px;
    background:rgba(255,255,255,0.12);
    border:1px solid rgba(255,255,255,0.2);
    border-radius:20px; padding:6px 16px;
    font-size:0.82rem; color:rgba(255,255,255,0.85); font-weight:500;
}

/* KPI */
.kpi-card {
    background:white; border-radius:14px; padding:22px 24px;
    box-shadow:0 2px 12px rgba(0,0,0,0.06);
    border-left:4px solid #1B4F72;
    transition:transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,0.1); }
.kpi-card.fog  { border-left-color:#C0392B; background:linear-gradient(135deg,#fff 70%,#FDEDEC); }
.kpi-card.norm { border-left-color:#148F77; background:linear-gradient(135deg,#fff 70%,#EAFAF1); }
.kpi-card.info { border-left-color:#1B4F72; background:linear-gradient(135deg,#fff 70%,#EBF5FB); }
.kpi-card.warn { border-left-color:#E67E22; background:linear-gradient(135deg,#fff 70%,#FEF9E7); }
.kpi-label { font-size:0.72rem; font-weight:600; color:#7F8C8D; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
.kpi-value { font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:#0A2342; line-height:1; margin-bottom:4px; }
.kpi-sub   { font-size:0.78rem; color:#95A5A6; }

/* SCORE DE RISQUE */
.score-container {
    background:white; border-radius:16px; padding:28px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08); text-align:center;
    margin-bottom:20px;
}
.score-circle {
    width:130px; height:130px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    margin:0 auto 16px auto;
    font-family:'Playfair Display',serif;
    font-size:2.2rem; font-weight:700; color:white;
    box-shadow:0 6px 20px rgba(0,0,0,0.2);
}
.score-low    { background:linear-gradient(135deg,#1A7A4A,#27AE60); }
.score-medium { background:linear-gradient(135deg,#E67E22,#F39C12); }
.score-high   { background:linear-gradient(135deg,#922B21,#C0392B); animation:pulse-red 2s infinite; }
.score-title  { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:600; color:#0A2342; }
.score-sub    { font-size:0.82rem; color:#7F8C8D; margin-top:4px; }

/* DIAGNOSTIC */
.diag-box { border-radius:14px; padding:24px; text-align:center; margin-bottom:16px; box-shadow:0 4px 16px rgba(0,0,0,0.08); }
.diag-box.fog     { background:linear-gradient(135deg,#C0392B,#E74C3C); animation:pulse-red 2s infinite; }
.diag-box.normal  { background:linear-gradient(135deg,#1A7A4A,#27AE60); }
.diag-box.waiting { background:linear-gradient(135deg,#2C3E50,#34495E); }
@keyframes pulse-red {
    0%,100% { box-shadow:0 4px 16px rgba(192,57,43,0.3); }
    50%      { box-shadow:0 4px 30px rgba(192,57,43,0.7); }
}
.diag-icon  { font-size:2.5rem; margin-bottom:8px; }
.diag-label { font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:white; margin:0; }
.diag-sub   { font-size:0.85rem; color:rgba(255,255,255,0.75); margin-top:4px; }

/* ALERTE */
.alert-severe {
    background:linear-gradient(135deg,#922B21,#C0392B);
    border-radius:12px; padding:18px 22px; color:white;
    margin-bottom:16px; animation:pulse-red 1.5s infinite;
}
.alert-title { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; }
.alert-sub   { font-size:0.83rem; opacity:0.85; margin-top:4px; }

/* SECTION */
.section-title {
    font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:600;
    color:#0A2342; margin:0 0 16px 0; padding-bottom:8px;
    border-bottom:2px solid #EBF5FB;
}

/* PROBA */
.proba-container { background:white; border-radius:12px; padding:18px 20px; box-shadow:0 2px 10px rgba(0,0,0,0.06); margin-bottom:14px; }
.proba-label { font-size:0.72rem; font-weight:600; color:#7F8C8D; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; }
.proba-bar-bg   { background:#ECF0F1; border-radius:6px; height:10px; overflow:hidden; margin-bottom:6px; }
.proba-bar-fill { height:100%; border-radius:6px; }
.proba-value    { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:700; }

/* BUTTONS */
.stButton > button {
    background:linear-gradient(135deg,#1B4F72,#148F77) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    padding:10px 24px !important;
}
.stButton > button:hover { opacity:0.85 !important; }

/* HIDE */
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
[data-testid="stToolbar"] { display:none; }
hr { border-color:rgba(255,255,255,0.1) !important; }

/* TABLE */
.hist-table {
    background:white; border-radius:14px; padding:20px;
    box-shadow:0 2px 12px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
for key, val in [('authenticated',False),('username',''),
                 ('running',False),('email_sent',False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# FONCTIONS ML
# ============================================
@st.cache_resource
def load_model():
    model  = joblib.load("models/model_fog_rf.pkl")
    scaler = joblib.load("models/scaler_fog.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_ok = True
except:
    model_ok = False

def butterworth_filter(signal, cutoff=3, fs=64, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, signal)

def extract_features(window, fs=64):
    features = []
    for axis in range(window.shape[1]):
        sig = window[:, axis]
        features += [np.mean(sig), np.std(sig), np.max(sig), np.min(sig),
                     np.max(sig)-np.min(sig), np.sqrt(np.mean(sig**2))]
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs    = np.fft.rfftfreq(len(sig), d=1/fs)
        features.append(np.sum(fft_vals))
        freeze  = fft_vals[(freqs>=3)  & (freqs<=8)]
        locomot = fft_vals[(freqs>=0.5)& (freqs<=3)]
        features.append(np.sum(freeze**2)/(np.sum(locomot**2)+1e-6))
    return np.array(features)

def predict_window(window, model, scaler):
    w = window.copy()
    for i in range(3): w[:,i] = butterworth_filter(w[:,i])
    feat   = extract_features(w).reshape(1,-1)
    feat_s = scaler.transform(feat)
    pred   = model.predict(feat_s)[0]
    proba  = model.predict_proba(feat_s)[0][1]
    return int(pred), round(float(proba)*100, 1)

# ============================================
# IDÉE 1 — SCORE DE RISQUE
# ============================================
def calcul_score_risque(cadence, variabilite, freeze_index, proba_fog, n_fog, duree):
    score = 0
    # Cadence anormale (< 90 pas/min)
    if cadence < 90:   score += 20
    elif cadence < 100: score += 10
    # Variabilité élevée
    if variabilite > 0.5:   score += 25
    elif variabilite > 0.2: score += 12
    # Freeze Index élevé
    if freeze_index > 3.0:  score += 20
    elif freeze_index > 1.5: score += 10
    # Probabilité FoG du modèle
    score += int(proba_fog * 0.20)
    # Fréquence des épisodes
    if duree > 0:
        freq = (n_fog / duree) * 60  # épisodes par minute
        if freq > 0.5:  score += 15
        elif freq > 0.2: score += 8
    return min(int(score), 100)

def get_niveau_risque(score):
    if score < 30:
        return "Faible", "#27AE60", "score-low", "✅"
    elif score < 65:
        return "Modéré", "#E67E22", "score-medium", "⚠️"
    else:
        return "Élevé",  "#C0392B", "score-high",   "🔴"

# ============================================
# IDÉE 2 — RAPPORT PDF
# ============================================
def generer_rapport_pdf(patient, age, sexe, date, duree,
                        n_fog, n_normal, score, cadence,
                        variabilite, fi, utilisateur):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    niveau, couleur, _, emoji = get_niveau_risque(score)

    # ---- EN-TÊTE ----
    pdf.set_fill_color(10, 35, 66)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(15, 10)
    pdf.cell(0, 10, "NeuroGait - Rapport d'Analyse", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(15, 24)
    pdf.cell(0, 8, "Systeme de Detection de la Maladie de Parkinson - PFE Ingenierie Biomedicale")

    # ---- INFOS PATIENT ----
    pdf.set_xy(15, 50)
    pdf.set_text_color(10, 35, 66)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Informations Patient", ln=True)
    pdf.set_draw_color(27, 79, 114)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(44, 62, 80)
    infos = [
        ("Nom du patient", patient),
        ("Age", f"{age} ans"),
        ("Sexe", sexe),
        ("Date de la session", date),
        ("Medecin / Utilisateur", utilisateur),
    ]
    for label, valeur in infos:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 7, f"{label} :", ln=False)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, valeur, ln=True)

    # ---- SCORE DE RISQUE ----
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(10, 35, 66)
    pdf.cell(0, 8, "Score de Risque Global", ln=True)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Barre de score
    pdf.set_fill_color(236, 240, 241)
    pdf.rect(15, pdf.get_y(), 180, 12, 'F')
    if score < 30:   pdf.set_fill_color(39, 174, 96)
    elif score < 65: pdf.set_fill_color(230, 126, 18)
    else:            pdf.set_fill_color(192, 57, 43)
    pdf.rect(15, pdf.get_y(), int(180 * score/100), 12, 'F')
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",10)
    pdf.set_xy(15, pdf.get_y()+1)
    pdf.cell(180, 10, f"  Score : {score}/100 — Risque {niveau}", ln=True)
    pdf.ln(2)

    pdf.set_text_color(44, 62, 80)
    pdf.set_font("Helvetica","",10)
    if score < 30:
        pdf.multi_cell(0, 6, "Interpretation : La marche du patient presente des parametres normaux. Aucune intervention urgente requise. Surveillance de routine recommandee.")
    elif score < 65:
        pdf.multi_cell(0, 6, "Interpretation : Des anomalies modérees ont été detectees. Une consultation neurologique est conseillée pour ajuster le traitement.")
    else:
        pdf.multi_cell(0, 6, "Interpretation : RISQUE ELEVE. Des episodes de Freezing of Gait severes ont ete detectes. Une consultation neurologique urgente est fortement recommandee.")

    # ---- BIOMARQUEURS ----
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(10, 35, 66)
    pdf.cell(0, 8, "Biomarqueurs de la Marche", ln=True)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)

    # Tableau
    headers = ["Biomarqueur", "Valeur Patient", "Norme Saine", "Statut"]
    col_w   = [65, 40, 45, 35]

    pdf.set_fill_color(27, 79, 114)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 10)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 8, h, border=1, fill=True)
    pdf.ln()

    rows = [
        ("Duree session",    f"{duree:.0f} s",     "—",        "—"),
        ("Cadence",          f"{cadence:.1f} ppm", "~110 ppm", "⚠" if cadence<90 else "OK"),
        ("Variabilite",      f"{variabilite:.3f} s","< 0.05 s", "⚠" if variabilite>0.2 else "OK"),
        ("Freeze Index",     f"{fi:.2f}",           "< 1.5",    "⚠" if fi>1.5 else "OK"),
        ("Episodes FoG",     str(n_fog),            "0",        "⚠" if n_fog>0 else "OK"),
        ("Fenetres Normales",str(n_normal),         "—",        "OK"),
    ]

    pdf.set_font("Helvetica", "", 10)
    for i, row in enumerate(rows):
        if i % 2 == 0: pdf.set_fill_color(245, 247, 250)
        else:          pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(44, 62, 80)
        for j, cell in enumerate(row):
            if j == 3 and cell == "⚠":
                pdf.set_text_color(192, 57, 43)
            elif j == 3:
                pdf.set_text_color(39, 174, 96)
            pdf.cell(col_w[j], 7, cell, border=1, fill=True)
            pdf.set_text_color(44, 62, 80)
        pdf.ln()

    # ---- PIED DE PAGE ----
    pdf.set_y(-30)
    pdf.set_draw_color(27, 79, 114)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(149, 165, 166)
    pdf.cell(0, 5, f"Genere le {date} par NeuroGait — PFE Ingenierie Biomedicale 2026", ln=True, align='C')
    pdf.cell(0, 5, "Ce rapport est genere automatiquement. Il ne remplace pas un diagnostic medical professionnel.", ln=True, align='C')

    return bytes(pdf.output())

# ============================================
# IDÉE 3 — ALERTE GMAIL
# ============================================
def envoyer_alerte_gmail(destinataire, patient, n_fog,
                         score, gmail_user, gmail_pass,
                         pdf_bytes=None):
    try:
        msg = MIMEMultipart()
        msg['From']    = gmail_user
        msg['To']      = destinataire
        msg['Subject'] = f"🚨 ALERTE NeuroGait — FoG Sévère détecté ({patient})"

        niveau, _, _, _ = get_niveau_risque(score)
        corps = f"""
Bonjour,

Le système NeuroGait a détecté une activité de marche critique.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DÉTAILS DE L'ALERTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Patient       : {patient}
Date          : {datetime.now().strftime("%d/%m/%Y à %H:%M")}
Épisodes FoG  : {n_fog} épisodes détectés
Score de risque : {score}/100 — Risque {niveau}

⚠️ Action recommandée : Consultation neurologique urgente.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Le rapport PDF complet est joint à cet email.

Cordialement,
Système NeuroGait — PFE Ingénierie Biomédicale 2026
        """
        msg.attach(MIMEText(corps, 'plain', 'utf-8'))

        # Joindre le PDF
        if pdf_bytes:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(pdf_bytes)
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                            f'attachment; filename="rapport_{patient}.pdf"')
            msg.attach(part)

        # Envoi via Gmail SMTP
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(gmail_user, gmail_pass)
        server.sendmail(gmail_user, destinataire, msg.as_string())
        server.quit()
        return True, "✅ Email envoyé avec succès !"
    except Exception as e:
        return False, f"❌ Erreur email : {str(e)}"

# ============================================
# PAGE LOGIN
# ============================================
if not st.session_state['authenticated']:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display:none !important; }
    [data-testid="stAppViewContainer"] > .main .block-container {
        padding:0 !important; max-width:100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;margin-bottom:8px;'>
            <div style='font-size:4rem;'>🧠</div>
            <div style='font-family:Playfair Display,serif;font-size:2rem;
                        font-weight:700;color:#0A2342;margin-top:8px;'>NeuroGait</div>
            <div style='font-size:0.85rem;color:#95A5A6;margin-top:6px;
                        margin-bottom:28px;line-height:1.5;'>
                Système de Détection de la Maladie de Parkinson<br>
                PFE — Ingénierie Biomédicale
            </div>
        </div>
        """, unsafe_allow_html=True)

        username = st.text_input("👤 Identifiant", placeholder="Entrez votre identifiant")
        password = st.text_input("🔒 Mot de passe", type="password", placeholder="Entrez votre mot de passe")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Se connecter →", use_container_width=True):
            if check_password(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.rerun()
            else:
                st.error("❌ Identifiant ou mot de passe incorrect")

        st.markdown("""
        <div style='text-align:center;margin-top:20px;font-size:0.78rem;
                    color:#BDC3C7;line-height:1.6;'>
            Accès réservé au personnel médical autorisé<br>
            © 2026 NeuroGait
        </div>
        """, unsafe_allow_html=True)

        with st.expander("💡 Comptes de démonstration"):
            st.markdown("""
            | Identifiant | Mot de passe |
            |---|---|
            | `admin` | `admin123` |
            | `medecin` | `parkinson2026` |
            | `lina` | `pfe2026` |
            """)
    st.stop()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:20px 0 10px 0;'>
        <div style='font-size:3rem;'>🧠</div>
        <div style='font-family:Playfair Display,serif;font-size:1.3rem;
                    font-weight:700;color:white;margin-top:8px;'>NeuroGait</div>
        <div style='font-size:0.72rem;color:rgba(255,255,255,0.45);
                    letter-spacing:2px;text-transform:uppercase;'>
            Analyse de Marche
        </div>
    </div>
    <hr>
    <div style='background:rgba(255,255,255,0.08);border-radius:10px;
                padding:10px 14px;margin-bottom:16px;'>
        <div style='font-size:0.7rem;color:rgba(255,255,255,0.4);
                    text-transform:uppercase;letter-spacing:1px;'>Connecté en tant que</div>
        <div style='font-size:0.95rem;font-weight:600;color:white;margin-top:2px;'>
            👤 {st.session_state['username']}
        </div>
    </div>
    <div style='font-size:0.7rem;letter-spacing:2px;color:rgba(255,255,255,0.4);
                text-transform:uppercase;margin-bottom:12px;'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📁 Analyse Fichier",
        "🎬 Démo Temps Réel",
        "📊 Historique Sessions",
        "⚙️ Configuration Alertes"
    ], label_visibility="collapsed")

    st.markdown("<hr><div style='font-size:0.7rem;letter-spacing:2px;color:rgba(255,255,255,0.4);text-transform:uppercase;margin-bottom:12px;'>Dossier Patient</div>", unsafe_allow_html=True)
    nom_patient = st.text_input("Nom", "Patient 001", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1: age  = st.number_input("Âge", 0, 120, 68, label_visibility="collapsed")
    with c2: sexe = st.selectbox("Sexe", ["H","F"], label_visibility="collapsed")

    st.markdown("<hr>", unsafe_allow_html=True)
    status = "✅ Modèle ML chargé" if model_ok else "⚠️ Modèle non trouvé"
    color  = "rgba(39,174,96,0.15)" if model_ok else "rgba(231,76,60,0.15)"
    border = "rgba(39,174,96,0.4)"  if model_ok else "rgba(231,76,60,0.4)"
    tcolor = "#7DCEA0" if model_ok else "#F1948A"
    st.markdown(f"<div style='background:{color};border:1px solid {border};border-radius:8px;padding:10px 14px;'><span style='color:{tcolor};font-size:0.8rem;font-weight:600;'>{status}</span></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ''
        st.rerun()

# ============================================
# HERO
# ============================================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-user">👤 {st.session_state['username']}</div>
    <div class="hero-badge">🔬 Système de Diagnostic Médical</div>
    <div class="hero-title">Analyse de la Marche Parkinsonienne</div>
    <div class="hero-subtitle">
        Détection FoG · Score de Risque · Rapport PDF · Alertes Automatiques
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# PAGE : ANALYSE FICHIER
# ============================================
if page == "📁 Analyse Fichier":

    st.markdown('<div class="section-title">📂 Chargement des Données IMU</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Fichier .txt (Daphnet) ou .csv",
                                type=["txt","csv"], label_visibility="collapsed")

    if uploaded:
        col_names = ['time','ankle_x','ankle_y','ankle_z',
                     'thigh_x','thigh_y','thigh_z',
                     'trunk_x','trunk_y','trunk_z','label']
        try:
            df = pd.read_csv(uploaded, sep=" ", header=None, names=col_names)
            df = df[df['label'] != 0].reset_index(drop=True)

            WINDOW_SIZE, OVERLAP = 128, 64
            windows_t, labels_pred, labels_real = [], [], []
            probas_list = []

            for start in range(0, len(df) - WINDOW_SIZE, OVERLAP):
                end    = start + WINDOW_SIZE
                window = df[['ankle_x','ankle_y','ankle_z']].iloc[start:end].values
                label  = df['label'].iloc[start:end].mode()[0]
                if model_ok:
                    pred, proba = predict_window(window, model, scaler)
                    labels_pred.append(pred)
                    probas_list.append(proba)
                labels_real.append(1 if label==2 else 0)
                windows_t.append(start/64)

            fog_pred = sum(labels_pred) if model_ok else 0
            fog_real = sum(labels_real)
            duree    = len(df)/64

            # Calcul biomarqueurs moyens
            az_vals    = df['ankle_z'].values
            cadence    = 60 / (np.mean(np.diff(np.where(np.diff(az_vals > np.mean(az_vals)))[0]))/64) if len(az_vals) > 128 else 100.0
            variabilite = np.std(np.diff(np.where(np.diff(az_vals > np.mean(az_vals)))[0])/64) if len(az_vals) > 128 else 0.1
            cadence     = min(max(cadence, 50), 150)
            variabilite = min(variabilite, 2.0)

            # Freeze Index moyen
            fft_vals = np.abs(np.fft.rfft(az_vals[:512]))
            freqs    = np.fft.rfftfreq(512, d=1/64)
            freeze   = fft_vals[(freqs>=3)&(freqs<=8)]
            locomot  = fft_vals[(freqs>=0.5)&(freqs<=3)]
            fi_moy   = float(np.sum(freeze**2)/(np.sum(locomot**2)+1e-6))

            proba_moy = np.mean(probas_list) * 100 if probas_list else 0

            # ---- SCORE DE RISQUE ----
            score = calcul_score_risque(cadence, variabilite, fi_moy,
                                        proba_moy/100, fog_pred, duree)
            niveau, couleur_score, score_class, emoji_score = get_niveau_risque(score)

            # ---- KPI ----
            k1,k2,k3,k4,k5 = st.columns(5)
            with k1: st.markdown(f'<div class="kpi-card info"><div class="kpi-label">⏱ Durée</div><div class="kpi-value">{duree:.0f}s</div><div class="kpi-sub">{len(df)} échantillons</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="kpi-card norm"><div class="kpi-label">✅ Normal</div><div class="kpi-value">{len(labels_pred)-fog_pred if labels_pred else 0}</div><div class="kpi-sub">fenêtres saines</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="kpi-card fog"><div class="kpi-label">🔴 FoG IA</div><div class="kpi-value">{fog_pred}</div><div class="kpi-sub">épisodes détectés</div></div>', unsafe_allow_html=True)
            with k4: st.markdown(f'<div class="kpi-card warn"><div class="kpi-label">📊 Cadence</div><div class="kpi-value">{cadence:.0f}</div><div class="kpi-sub">pas/min</div></div>', unsafe_allow_html=True)
            with k5:
                sc = "fog" if score>=65 else ("warn" if score>=30 else "norm")
                st.markdown(f'<div class="kpi-card {sc}"><div class="kpi-label">{emoji_score} Score Risque</div><div class="kpi-value">{score}</div><div class="kpi-sub">Risque {niveau}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---- LAYOUT PRINCIPAL ----
            col_main, col_score = st.columns([3, 1])

            with col_score:
                # SCORE DE RISQUE VISUEL
                st.markdown(f"""
                <div class="score-container">
                    <div class="section-title" style="text-align:center;">
                        🎯 Score de Risque
                    </div>
                    <div class="score-circle {score_class}">
                        {score}
                    </div>
                    <div class="score-title">{emoji_score} Risque {niveau}</div>
                    <div class="score-sub">/100 — Session du {datetime.now().strftime("%d/%m/%Y")}</div>
                </div>
                """, unsafe_allow_html=True)

                # Détail du score
                st.markdown(f"""
                <div style='background:white;border-radius:12px;padding:16px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
                    <div class="kpi-label" style="margin-bottom:12px;">Détail des biomarqueurs</div>
                    <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                        <span style='font-size:0.82rem;color:#2C3E50;'>Cadence</span>
                        <span style='font-size:0.82rem;font-weight:600;
                              color:{"#C0392B" if cadence<90 else "#27AE60"};'>
                            {cadence:.0f} ppm
                        </span>
                    </div>
                    <div style='display:flex;justify-content:space-between;margin-bottom:8px;'>
                        <span style='font-size:0.82rem;color:#2C3E50;'>Variabilité</span>
                        <span style='font-size:0.82rem;font-weight:600;
                              color:{"#C0392B" if variabilite>0.2 else "#27AE60"};'>
                            {variabilite:.3f} s
                        </span>
                    </div>
                    <div style='display:flex;justify-content:space-between;'>
                        <span style='font-size:0.82rem;color:#2C3E50;'>Freeze Index</span>
                        <span style='font-size:0.82rem;font-weight:600;
                              color:{"#C0392B" if fi_moy>1.5 else "#27AE60"};'>
                            {fi_moy:.2f}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_main:
                st.markdown('<div class="section-title">📈 Signal Accéléromètre (Axe Z)</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df['ankle_z'].values[:3000], mode='lines',
                    line=dict(color='#1B4F72', width=1.2)))
                in_fog=False; sf=0
                for i in range(min(3000,len(df))):
                    if df['label'].iloc[i]==2 and not in_fog: sf=i; in_fog=True
                    elif df['label'].iloc[i]!=2 and in_fog:
                        fig.add_vrect(x0=sf,x1=i,fillcolor="#E74C3C",opacity=0.12,line_width=0)
                        in_fog=False
                fig.update_layout(height=250,margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor='white',paper_bgcolor='white',
                    font=dict(color='#2C3E50',family='DM Sans'),
                    xaxis=dict(title="Échantillons",showgrid=True,gridcolor='#F5F5F5'),
                    yaxis=dict(title="Accélération (mg)",showgrid=True,gridcolor='#F5F5F5'),
                    showlegend=False)
                st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

                if model_ok and probas_list:
                    st.markdown('<div class="section-title">🤖 Probabilité FoG dans le Temps</div>', unsafe_allow_html=True)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=windows_t, y=probas_list, mode='lines',
                        fill='tozeroy',
                        line=dict(color='#E74C3C', width=1.5),
                        fillcolor='rgba(231,76,60,0.1)'
                    ))
                    fig2.add_hline(y=60, line_dash="dash",
                                   line_color="#E67E22", line_width=1.5,
                                   annotation_text="Seuil 60%")
                    fig2.update_layout(height=200,margin=dict(l=10,r=10,t=10,b=10),
                        plot_bgcolor='white',paper_bgcolor='white',
                        font=dict(color='#2C3E50',family='DM Sans'),
                        xaxis=dict(title="Temps (s)",showgrid=True,gridcolor='#F5F5F5'),
                        yaxis=dict(title="%",showgrid=True,gridcolor='#F5F5F5',range=[0,105]),
                        showlegend=False)
                    st.plotly_chart(fig2,use_container_width=True,config={'displayModeBar':False})

            # ---- ACTIONS : PDF + EMAIL ----
            st.markdown("---")
            st.markdown('<div class="section-title">📄 Actions</div>', unsafe_allow_html=True)
            btn1, btn2 = st.columns(2)

            # Générer PDF
            pdf_bytes = generer_rapport_pdf(
                nom_patient, age, sexe,
                datetime.now().strftime("%d/%m/%Y %H:%M"),
                duree, fog_pred,
                len(labels_pred)-fog_pred if labels_pred else 0,
                score, cadence, variabilite, fi_moy,
                st.session_state['username']
            )

            with btn1:
                st.download_button(
                    label="📥 Télécharger Rapport PDF",
                    data=pdf_bytes,
                    file_name=f"rapport_{nom_patient}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            with btn2:
                if st.button("💾 Sauvegarder dans l'historique",
                             use_container_width=True):
                    save_session(nom_patient, age, sexe, duree,
                                 fog_pred,
                                 len(labels_pred)-fog_pred if labels_pred else 0,
                                 score, cadence, variabilite, fi_moy,
                                 st.session_state['username'])
                    st.success("✅ Session sauvegardée dans l'historique !")

            # Alerte si risque élevé
            if score >= 65 and fog_pred > 0:
                st.markdown(f"""
                <div class="alert-severe">
                    <div class="alert-title">🚨 Risque Élevé Détecté</div>
                    <div class="alert-sub">
                        {fog_pred} épisodes FoG détectés — Score : {score}/100<br>
                        Configure les alertes email dans ⚙️ Configuration Alertes
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    else:
        st.markdown("""
        <div style='background:white;border-radius:14px;padding:56px;
                    text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.06);'>
            <div style='font-size:3.5rem;margin-bottom:16px;'>📂</div>
            <div style='font-family:Playfair Display,serif;font-size:1.4rem;
                        font-weight:600;color:#0A2342;margin-bottom:10px;'>
                Aucun fichier chargé
            </div>
            <div style='color:#95A5A6;font-size:0.92rem;max-width:380px;
                        margin:0 auto;line-height:1.6;'>
                Glisse un fichier <b>.txt</b> du dataset Daphnet FoG
                pour démarrer l'analyse.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE : DÉMO TEMPS RÉEL
# ============================================
elif page == "🎬 Démo Temps Réel":

    col_chart, col_diag = st.columns([3,1])

    with col_diag:
        st.markdown('<div class="section-title">🔬 Diagnostic</div>', unsafe_allow_html=True)
        diag_box   = st.empty()
        proba_box  = st.empty()
        score_box  = st.empty()
        stats_box  = st.empty()
        alert_box  = st.empty()
        diag_box.markdown('<div class="diag-box waiting"><div class="diag-icon">⏳</div><div class="diag-label">En attente</div><div class="diag-sub">Cliquez Démarrer</div></div>', unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="section-title">📈 Signal IMU — Temps Réel</div>', unsafe_allow_html=True)
        chart_box = st.empty()

    c1,c2,_ = st.columns([1,1,5])
    with c1: start_btn = st.button("▶️ Démarrer",type="primary",use_container_width=True)
    with c2: stop_btn  = st.button("⏹️ Arrêter", use_container_width=True)

    if start_btn: st.session_state['running']=True
    if stop_btn:  st.session_state['running']=False

    if st.session_state.get('running', False):
        buffer=collections.deque(maxlen=128)
        hist_ax,hist_ay,hist_az=[],[],[]
        n_fog=n_normal=t=0
        last_pred=0; last_proba=0.0
        session_scores=[]

        while st.session_state.get('running',False):
            is_fog=(t%320>230)
            if is_fog:
                ax=np.random.normal(300,90); ay=np.random.normal(950,130)
                az=np.random.normal(100,210)+160*np.sin(2*np.pi*5*t/64)
            else:
                ax=np.random.normal(50,20); ay=np.random.normal(980,30)
                az=np.random.normal(-80,40)+65*np.sin(2*np.pi*1.8*t/64)

            buffer.append([ax,ay,az])
            hist_ax.append(ax); hist_ay.append(ay); hist_az.append(az)
            if len(hist_ax)>320:
                hist_ax=hist_ax[-320:]; hist_ay=hist_ay[-320:]; hist_az=hist_az[-320:]

            if len(buffer)==128 and model_ok:
                last_pred,last_proba=predict_window(np.array(buffer),model,scaler)
                if last_pred==1: n_fog+=1
                else: n_normal+=1

                # Score de risque dynamique
                duree_s = t/64
                score_dyn = calcul_score_risque(
                    cadence=95 if is_fog else 108,
                    variabilite=0.8 if is_fog else 0.03,
                    freeze_index=3.5 if is_fog else 0.8,
                    proba_fog=last_proba/100,
                    n_fog=n_fog, duree=max(duree_s,1)
                )
                niveau_d, _, sc_cls, emoji_d = get_niveau_risque(score_dyn)

                # Score box
                score_box.markdown(f"""
                <div class="score-container" style="padding:16px;">
                    <div class="score-circle {sc_cls}" style="width:80px;height:80px;font-size:1.4rem;">
                        {score_dyn}
                    </div>
                    <div class="score-title" style="font-size:0.9rem;">
                        {emoji_d} Risque {niveau_d}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Alerte sévère
                if n_fog >= 3:
                    alert_box.markdown(f"""
                    <div class="alert-severe">
                        <div class="alert-title">🚨 FoG Sévère !</div>
                        <div class="alert-sub">{n_fog} épisodes · Score {score_dyn}/100</div>
                    </div>
                    """, unsafe_allow_html=True)

            if last_pred==1:
                diag_box.markdown('<div class="diag-box fog"><div class="diag-icon">🔴</div><div class="diag-label">FoG DÉTECTÉ !</div><div class="diag-sub">Enrayement cinétique</div></div>', unsafe_allow_html=True)
            else:
                diag_box.markdown('<div class="diag-box normal"><div class="diag-icon">🟢</div><div class="diag-label">Marche Normale</div><div class="diag-sub">Aucune anomalie</div></div>', unsafe_allow_html=True)

            bc="#E74C3C" if last_proba>60 else "#27AE60"
            proba_box.markdown(f'<div class="proba-container"><div class="proba-label">Probabilité FoG</div><div class="proba-bar-bg"><div class="proba-bar-fill" style="width:{last_proba}%;background:{bc};"></div></div><div class="proba-value" style="color:{bc};">{last_proba}%</div></div>', unsafe_allow_html=True)
            stats_box.markdown(f'<div style="background:white;border-radius:12px;padding:14px;box-shadow:0 2px 8px rgba(0,0,0,0.05);"><div style="font-size:0.7rem;font-weight:600;color:#7F8C8D;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Stats</div><div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="color:#27AE60;font-weight:600;font-size:0.85rem;">🟢 Normal</span><span style="font-weight:700;color:#0A2342;">{n_normal}</span></div><div style="display:flex;justify-content:space-between;"><span style="color:#E74C3C;font-weight:600;font-size:0.85rem;">🔴 FoG</span><span style="font-weight:700;color:#E74C3C;">{n_fog}</span></div></div>', unsafe_allow_html=True)

            lc='#E74C3C' if is_fog else '#1B4F72'
            fig=go.Figure()
            fig.add_trace(go.Scatter(y=hist_ax,mode='lines',name='X',line=dict(color='#AED6F1',width=1)))
            fig.add_trace(go.Scatter(y=hist_ay,mode='lines',name='Y',line=dict(color='#A9DFBF',width=1)))
            fig.add_trace(go.Scatter(y=hist_az,mode='lines',name='Z',line=dict(color=lc,width=2)))
            fig.update_layout(height=320,margin=dict(l=10,r=10,t=10,b=10),
                plot_bgcolor='white',paper_bgcolor='white',
                font=dict(color='#2C3E50',family='DM Sans'),
                xaxis=dict(showgrid=True,gridcolor='#F5F5F5'),
                yaxis=dict(title='Accélération (mg)',showgrid=True,gridcolor='#F5F5F5'),
                legend=dict(orientation='h',yanchor='bottom',y=1.02,font=dict(size=11)))
            chart_box.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
            t+=1; time.sleep(0.04)

# ============================================
# PAGE : HISTORIQUE
# ============================================
elif page == "📊 Historique Sessions":
    st.markdown('<div class="section-title">📊 Historique des Sessions</div>', unsafe_allow_html=True)
    try:
        df_hist = get_sessions()
        if len(df_hist) == 0:
            st.info("Aucune session enregistrée. Faites une analyse et cliquez sur 'Sauvegarder'.")
        else:
            # Stats globales
            h1,h2,h3,h4 = st.columns(4)
            with h1: st.markdown(f'<div class="kpi-card info"><div class="kpi-label">📋 Total Sessions</div><div class="kpi-value">{len(df_hist)}</div><div class="kpi-sub">enregistrées</div></div>', unsafe_allow_html=True)
            with h2: st.markdown(f'<div class="kpi-card fog"><div class="kpi-label">🔴 Total FoG</div><div class="kpi-value">{df_hist["fog_episodes"].sum()}</div><div class="kpi-sub">épisodes cumulés</div></div>', unsafe_allow_html=True)
            with h3: st.markdown(f'<div class="kpi-card warn"><div class="kpi-label">⚠️ Score Moyen</div><div class="kpi-value">{df_hist["score_risque"].mean():.0f}</div><div class="kpi-sub">/100</div></div>', unsafe_allow_html=True)
            with h4: st.markdown(f'<div class="kpi-card norm"><div class="kpi-label">👥 Patients</div><div class="kpi-value">{df_hist["patient"].nunique()}</div><div class="kpi-sub">uniques</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Évolution scores
            st.markdown('<div class="section-title">📈 Évolution du Score de Risque</div>', unsafe_allow_html=True)
            fig_hist = go.Figure()
            for patient in df_hist['patient'].unique():
                df_p = df_hist[df_hist['patient']==patient]
                fig_hist.add_trace(go.Scatter(
                    x=df_p['date'], y=df_p['score_risque'],
                    mode='lines+markers', name=patient,
                    line=dict(width=2)))
            fig_hist.add_hline(y=65, line_dash="dash",
                               line_color="#C0392B", line_width=1.5,
                               annotation_text="Seuil Risque Élevé")
            fig_hist.update_layout(height=280,
                plot_bgcolor='white',paper_bgcolor='white',
                font=dict(color='#2C3E50',family='DM Sans'),
                xaxis=dict(title="Date",showgrid=True,gridcolor='#F5F5F5'),
                yaxis=dict(title="Score de Risque",showgrid=True,
                           gridcolor='#F5F5F5',range=[0,105]),
                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_hist,use_container_width=True,config={'displayModeBar':False})

            # Tableau
            st.markdown('<div class="section-title">📋 Détail des Sessions</div>', unsafe_allow_html=True)
            cols_show = ['patient','age','sexe','date','duree',
                         'fog_episodes','score_risque','utilisateur']
            st.dataframe(df_hist[cols_show].rename(columns={
                'patient':'Patient','age':'Âge','sexe':'Sexe',
                'date':'Date','duree':'Durée (s)',
                'fog_episodes':'Épisodes FoG',
                'score_risque':'Score Risque',
                'utilisateur':'Utilisateur'
            }), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Erreur historique : {e}")

# ============================================
# PAGE : CONFIGURATION ALERTES
# ============================================
elif page == "⚙️ Configuration Alertes":
    st.markdown('<div class="section-title">⚙️ Configuration des Alertes Email</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#EBF5FB;border:1px solid #1B4F72;border-radius:10px;
                padding:16px 20px;margin-bottom:20px;'>
        <b style='color:#1B4F72;'>ℹ️ Comment configurer Gmail ?</b><br>
        <div style='color:#2C3E50;font-size:0.88rem;margin-top:6px;line-height:1.7;'>
            1. Active la <b>validation en 2 étapes</b> sur ton compte Gmail<br>
            2. Va dans <b>Paramètres → Sécurité → Mots de passe des applications</b><br>
            3. Génère un mot de passe pour "Mail" → copie les 16 caractères<br>
            4. Utilise ce mot de passe ci-dessous (pas ton mot de passe Gmail habituel)
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cfg1, col_cfg2 = st.columns(2)

    with col_cfg1:
        st.subheader("📤 Expéditeur (ton Gmail)")
        gmail_user = st.text_input("Adresse Gmail",
                                   placeholder="ton.email@gmail.com")
        gmail_pass = st.text_input("Mot de passe application (16 car.)",
                                   type="password",
                                   placeholder="xxxx xxxx xxxx xxxx")

    with col_cfg2:
        st.subheader("📬 Destinataire (médecin)")
        dest_email = st.text_input("Email du médecin / destinataire",
                                   placeholder="medecin@hopital.ma")
        seuil_fog  = st.slider("Envoyer alerte si FoG ≥",
                               min_value=1, max_value=10, value=3)
        seuil_score = st.slider("Envoyer alerte si Score ≥",
                                min_value=30, max_value=100, value=65)

    st.markdown("<br>", unsafe_allow_html=True)

    # Test d'envoi
    if st.button("📧 Envoyer un Email de Test", use_container_width=False):
        if not gmail_user or not gmail_pass or not dest_email:
            st.error("❌ Remplis tous les champs avant d'envoyer !")
        else:
            # PDF de test
            pdf_test = generer_rapport_pdf(
                "Patient Test", 65, "H",
                datetime.now().strftime("%d/%m/%Y %H:%M"),
                120, 5, 45, 72, 88.5, 0.45, 2.8,
                st.session_state['username']
            )
            ok, msg = envoyer_alerte_gmail(
                dest_email, "Patient Test",
                5, 72, gmail_user, gmail_pass, pdf_test
            )
            if ok: st.success(msg)
            else:  st.error(msg)

    st.markdown("---")

    # Envoi manuel pour un patient
    st.subheader("📨 Envoyer une Alerte Manuelle")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        patient_alert = st.text_input("Nom du patient", nom_patient)
        fog_alert     = st.number_input("Nombre d'épisodes FoG", 0, 100, 5)
    with col_m2:
        score_alert   = st.number_input("Score de risque", 0, 100, 72)

    if st.button("🚨 Envoyer l'Alerte", type="primary"):
        if not gmail_user or not gmail_pass or not dest_email:
            st.error("❌ Configure d'abord les paramètres Gmail ci-dessus !")
        else:
            pdf_alert = generer_rapport_pdf(
                patient_alert, age, sexe,
                datetime.now().strftime("%d/%m/%Y %H:%M"),
                0, fog_alert, 0, score_alert,
                100.0, 0.5, 2.0,
                st.session_state['username']
            )
            ok, msg = envoyer_alerte_gmail(
                dest_email, patient_alert,
                fog_alert, score_alert,
                gmail_user, gmail_pass, pdf_alert
            )
            if ok: st.success(msg)
            else:  st.error(msg)
