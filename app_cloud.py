# ============================================
# app_cloud.py — NeuroGait
# Système de Détection Parkinson
# Avec Authentification + Design Professionnel
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
 
st.set_page_config(
    page_title="NeuroGait — Détection Parkinson",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ============================================
# UTILISATEURS (mot de passe hashé SHA256)
# ============================================
# Pour ajouter un utilisateur : hashlib.sha256("motdepasse".encode()).hexdigest()
USERS = {
    "admin":   hashlib.sha256("admin123".encode()).hexdigest(),
    "medecin": hashlib.sha256("parkinson2026".encode()).hexdigest(),
    "lina":    hashlib.sha256("pfe2026".encode()).hexdigest(),
}
 
def check_password(username, password):
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == hashed
 
# ============================================
# CSS GLOBAL
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
 
/* ---- SIDEBAR ---- */
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
 
/* ---- LOGIN PAGE ---- */
.login-wrapper {
    min-height: 100vh;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #0A2342 0%, #1B4F72 50%, #148F77 100%);
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    z-index: 999;
}
.login-card {
    background: white;
    border-radius: 20px;
    padding: 48px 44px;
    width: 420px;
    box-shadow: 0 24px 64px rgba(0,0,0,0.25);
    text-align: center;
}
.login-logo { font-size: 3.5rem; margin-bottom: 8px; }
.login-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem; font-weight: 700;
    color: #0A2342; margin-bottom: 4px;
}
.login-subtitle {
    font-size: 0.85rem; color: #95A5A6;
    margin-bottom: 32px; line-height: 1.5;
}
.login-label {
    font-size: 0.75rem; font-weight: 600;
    color: #7F8C8D; text-transform: uppercase;
    letter-spacing: 1px; text-align: left;
    display: block; margin-bottom: 6px;
}
.login-divider {
    height: 1px; background: #ECF0F1;
    margin: 24px 0;
}
.login-footer {
    font-size: 0.72rem; color: #BDC3C7;
    margin-top: 24px; line-height: 1.6;
}
.error-box {
    background: #FDEDEC; border: 1px solid #E74C3C;
    border-radius: 8px; padding: 10px 14px;
    color: #C0392B; font-size: 0.85rem;
    font-weight: 500; margin-bottom: 16px;
    text-align: left;
}
 
/* ---- HERO ---- */
.hero-container {
    background: linear-gradient(135deg, #0A2342 0%, #1B4F72 55%, #148F77 100%);
    border-radius: 16px; padding: 36px 40px;
    margin-bottom: 28px; position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(10,35,66,0.25);
}
.hero-container::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background:rgba(255,255,255,0.04);
}
.hero-badge {
    display:inline-block;
    background:rgba(20,143,119,0.3);
    border:1px solid rgba(20,143,119,0.6);
    color:#7DCEA0; font-size:0.75rem; font-weight:600;
    letter-spacing:1.5px; text-transform:uppercase;
    padding:4px 12px; border-radius:20px; margin-bottom:14px;
}
.hero-title {
    font-family:'Playfair Display',serif;
    font-size:2.2rem; font-weight:700;
    color:#FFFFFF; margin:0 0 6px 0; letter-spacing:-0.5px;
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
    font-size:0.82rem; color:rgba(255,255,255,0.85);
    font-weight:500;
}
 
/* ---- KPI CARDS ---- */
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
.kpi-label { font-size:0.72rem; font-weight:600; color:#7F8C8D; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; }
.kpi-value { font-family:'Playfair Display',serif; font-size:2rem; font-weight:700; color:#0A2342; line-height:1; margin-bottom:4px; }
.kpi-sub   { font-size:0.78rem; color:#95A5A6; }
 
/* ---- DIAGNOSTIC ---- */
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
 
/* ---- PROBA BAR ---- */
.proba-container { background:white; border-radius:12px; padding:18px 20px; box-shadow:0 2px 10px rgba(0,0,0,0.06); margin-bottom:14px; }
.proba-label { font-size:0.72rem; font-weight:600; color:#7F8C8D; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; }
.proba-bar-bg   { background:#ECF0F1; border-radius:6px; height:10px; overflow:hidden; margin-bottom:6px; }
.proba-bar-fill { height:100%; border-radius:6px; transition:width 0.5s ease; }
.proba-value    { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:700; }
 
/* ---- SECTION TITLE ---- */
.section-title {
    font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:600;
    color:#0A2342; margin:0 0 16px 0; padding-bottom:8px;
    border-bottom:2px solid #EBF5FB;
}
 
/* ---- BUTTONS ---- */
.stButton > button {
    background:linear-gradient(135deg,#1B4F72,#148F77) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-family:'DM Sans',sans-serif !important; font-weight:600 !important;
    padding:10px 24px !important; transition:opacity 0.2s !important;
}
.stButton > button:hover { opacity:0.85 !important; }
 
/* ---- HIDE DEFAULTS ---- */
#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
[data-testid="stToolbar"] { display:none; }
hr { border-color:rgba(255,255,255,0.1) !important; }
</style>
""", unsafe_allow_html=True)
 
# ============================================
# SESSION STATE
# ============================================
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'running' not in st.session_state:
    st.session_state['running'] = False
 
# ============================================
# PAGE LOGIN
# ============================================
if not st.session_state['authenticated']:
 
    # Cache le sidebar sur la page login
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stAppViewContainer"] > .main .block-container {
        padding: 0 !important; max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
 
    # Centrage de la card
    _, col, _ = st.columns([1, 1.2, 1])
 
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
 
        # Logo + Titre
        st.markdown("""
        <div style='text-align:center; margin-bottom:8px;'>
            <div style='font-size:4rem;'>🧠</div>
            <div style='font-family:Playfair Display,serif; font-size:2rem;
                        font-weight:700; color:#0A2342; margin-top:8px;'>
                NeuroGait
            </div>
            <div style='font-size:0.85rem; color:#95A5A6; margin-top:6px;
                        margin-bottom:28px; line-height:1.5;'>
                Système de Détection de la Maladie de Parkinson<br>
                PFE — Ingénierie Biomédicale
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # Card de connexion
        with st.container():
            st.markdown("""
            <div style='background:white; border-radius:16px; padding:36px 32px;
                        box-shadow:0 16px 48px rgba(10,35,66,0.15);
                        border:1px solid #ECF0F1;'>
                <div style='font-family:Playfair Display,serif; font-size:1.3rem;
                            font-weight:600; color:#0A2342; margin-bottom:24px;
                            text-align:center;'>
                    Connexion
                </div>
            </div>
            """, unsafe_allow_html=True)
 
            username = st.text_input(
                "👤 Identifiant",
                placeholder="Entrez votre identifiant"
            )
            password = st.text_input(
                "🔒 Mot de passe",
                type="password",
                placeholder="Entrez votre mot de passe"
            )
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            if st.button("Se connecter →", use_container_width=True):
                if check_password(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("❌ Identifiant ou mot de passe incorrect")
 
            st.markdown("""
            <div style='text-align:center; margin-top:20px; font-size:0.78rem;
                        color:#BDC3C7; line-height:1.6;'>
                Accès réservé au personnel médical autorisé<br>
                © 2026 NeuroGait — Ingénierie Biomédicale
            </div>
            """, unsafe_allow_html=True)
 
        # Comptes de démonstration
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
# CHARGEMENT MODÈLE (après login)
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
 
# ============================================
# FONCTIONS ML
# ============================================
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
# SIDEBAR (après login)
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
                text-transform:uppercase;margin-bottom:12px;'>Mode</div>
    """, unsafe_allow_html=True)
 
    mode = st.radio("", ["📁 Analyse Fichier", "🎬 Démo Temps Réel"],
                    label_visibility="collapsed")
 
    st.markdown("<hr><div style='font-size:0.7rem;letter-spacing:2px;color:rgba(255,255,255,0.4);text-transform:uppercase;margin-bottom:12px;'>Dossier Patient</div>", unsafe_allow_html=True)
 
    nom_patient = st.text_input("Nom", "Patient 001", label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1: age  = st.number_input("Âge", 0, 120, 68, label_visibility="collapsed")
    with c2: sexe = st.selectbox("Sexe", ["H","F"], label_visibility="collapsed")
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    if model_ok:
        st.markdown("<div style='background:rgba(39,174,96,0.15);border:1px solid rgba(39,174,96,0.4);border-radius:8px;padding:10px 14px;'><span style='color:#7DCEA0;font-size:0.8rem;font-weight:600;'>✅ Modèle ML chargé</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:rgba(231,76,60,0.15);border:1px solid rgba(231,76,60,0.4);border-radius:8px;padding:10px 14px;'><span style='color:#F1948A;font-size:0.8rem;font-weight:600;'>⚠️ Modèle non trouvé</span></div>", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # Bouton déconnexion
    if st.button("🚪 Se déconnecter", use_container_width=True):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ''
        st.rerun()
 
# ============================================
# HERO HEADER
# ============================================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-user">👤 {st.session_state['username']}</div>
    <div class="hero-badge">🔬 Système de Diagnostic Médical</div>
    <div class="hero-title">Analyse de la Marche Parkinsonienne</div>
    <div class="hero-subtitle">
        Détection automatique du Freezing of Gait &nbsp;·&nbsp;
        Capteurs IMU &nbsp;·&nbsp; Intelligence Artificielle &nbsp;·&nbsp; Temps Réel
    </div>
</div>
""", unsafe_allow_html=True)
 
# ============================================
# MODE 1 — ANALYSE FICHIER
# ============================================
if mode == "📁 Analyse Fichier":
 
    st.markdown('<div class="section-title">📂 Chargement des Données IMU</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Fichier .txt ou .csv", type=["txt","csv"], label_visibility="collapsed")
 
    if uploaded:
        col_names = ['time','ankle_x','ankle_y','ankle_z',
                     'thigh_x','thigh_y','thigh_z',
                     'trunk_x','trunk_y','trunk_z','label']
        try:
            df = pd.read_csv(uploaded, sep=" ", header=None, names=col_names)
            df = df[df['label'] != 0].reset_index(drop=True)
 
            WINDOW_SIZE, OVERLAP = 128, 64
            windows_t, labels_pred, labels_real = [], [], []
 
            for start in range(0, len(df) - WINDOW_SIZE, OVERLAP):
                end    = start + WINDOW_SIZE
                window = df[['ankle_x','ankle_y','ankle_z']].iloc[start:end].values
                label  = df['label'].iloc[start:end].mode()[0]
                if model_ok:
                    pred, proba = predict_window(window, model, scaler)
                    labels_pred.append(pred)
                labels_real.append(1 if label==2 else 0)
                windows_t.append(start/64)
 
            fog_pred = sum(labels_pred) if model_ok else 0
            fog_real = sum(labels_real)
            duree    = len(df)/64
 
            # KPI
            k1,k2,k3,k4 = st.columns(4)
            with k1: st.markdown(f'<div class="kpi-card info"><div class="kpi-label">⏱ Durée totale</div><div class="kpi-value">{duree:.0f}s</div><div class="kpi-sub">{len(df)} échantillons · 64 Hz</div></div>', unsafe_allow_html=True)
            with k2: st.markdown(f'<div class="kpi-card norm"><div class="kpi-label">✅ Marche normale</div><div class="kpi-value">{len(df[df["label"]==1])}</div><div class="kpi-sub">échantillons sains</div></div>', unsafe_allow_html=True)
            with k3: st.markdown(f'<div class="kpi-card fog"><div class="kpi-label">🔴 FoG annotés</div><div class="kpi-value">{fog_real}</div><div class="kpi-sub">fenêtres pathologiques</div></div>', unsafe_allow_html=True)
            with k4:
                cls = "fog" if fog_pred>0 else "norm"
                st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-label">🤖 FoG détectés IA</div><div class="kpi-value">{fog_pred}</div><div class="kpi-sub">prédictions modèle</div></div>', unsafe_allow_html=True)
 
            st.markdown("<br>", unsafe_allow_html=True)
 
            cg1, cg2 = st.columns([3,2])
            with cg1:
                st.markdown('<div class="section-title">📈 Signal Accéléromètre Cheville (Axe Z)</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df['ankle_z'].values[:3000], mode='lines',
                    line=dict(color='#1B4F72', width=1.2)))
                in_fog=False; sf=0
                for i in range(min(3000,len(df))):
                    if df['label'].iloc[i]==2 and not in_fog: sf=i; in_fog=True
                    elif df['label'].iloc[i]!=2 and in_fog:
                        fig.add_vrect(x0=sf,x1=i,fillcolor="#E74C3C",opacity=0.12,line_width=0)
                        in_fog=False
                fig.update_layout(height=280,margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor='white',paper_bgcolor='white',
                    font=dict(color='#2C3E50',family='DM Sans'),
                    xaxis=dict(title="Échantillons",showgrid=True,gridcolor='#F5F5F5'),
                    yaxis=dict(title="Accélération (mg)",showgrid=True,gridcolor='#F5F5F5'),
                    showlegend=False)
                st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
 
            with cg2:
                st.markdown('<div class="section-title">🥧 Distribution des Classes</div>', unsafe_allow_html=True)
                n_norm=len(df[df['label']==1]); n_fog_=len(df[df['label']==2])
                fig_pie=go.Figure(go.Pie(
                    labels=['Marche Normale','Freezing of Gait'],
                    values=[n_norm,n_fog_],hole=0.55,
                    marker=dict(colors=['#1B4F72','#E74C3C'],line=dict(color='white',width=2)),
                    textfont=dict(family='DM Sans',size=12)))
                fig_pie.update_layout(height=280,margin=dict(l=10,r=10,t=10,b=10),
                    paper_bgcolor='white',font=dict(color='#2C3E50',family='DM Sans'),
                    legend=dict(orientation='h',yanchor='bottom',y=-0.15),
                    annotations=[dict(text=f'<b>{n_fog_/(n_norm+n_fog_)*100:.1f}%</b><br>FoG',
                        x=0.5,y=0.5,font=dict(size=13,color='#E74C3C'),showarrow=False)])
                st.plotly_chart(fig_pie,use_container_width=True,config={'displayModeBar':False})
 
            if model_ok and labels_pred:
                st.markdown('<div class="section-title">🤖 Timeline des Prédictions IA</div>', unsafe_allow_html=True)
                fig3=go.Figure()
                fig3.add_trace(go.Bar(x=windows_t,y=labels_pred,width=1.5,
                    marker_color=['#E74C3C' if p==1 else '#27AE60' for p in labels_pred]))
                fig3.update_layout(height=160,margin=dict(l=10,r=10,t=10,b=10),
                    plot_bgcolor='white',paper_bgcolor='white',
                    font=dict(color='#2C3E50',family='DM Sans'),
                    xaxis=dict(title="Temps (secondes)",showgrid=False),
                    yaxis=dict(visible=False,range=[-0.1,1.3]),showlegend=False)
                st.plotly_chart(fig3,use_container_width=True,config={'displayModeBar':False})
 
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
                Glisse un fichier <b>.txt</b> du dataset Daphnet FoG pour
                démarrer l'analyse complète.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
# ============================================
# MODE 2 — DÉMO TEMPS RÉEL
# ============================================
else:
    col_chart, col_diag = st.columns([3,1])
 
    with col_diag:
        st.markdown('<div class="section-title">🔬 Diagnostic</div>', unsafe_allow_html=True)
        diag_box  = st.empty()
        proba_box = st.empty()
        stats_box = st.empty()
        diag_box.markdown('<div class="diag-box waiting"><div class="diag-icon">⏳</div><div class="diag-label">En attente</div><div class="diag-sub">Cliquez Démarrer</div></div>', unsafe_allow_html=True)
 
    with col_chart:
        st.markdown('<div class="section-title">📈 Signal IMU — Temps Réel</div>', unsafe_allow_html=True)
        chart_box = st.empty()
 
    c1,c2,_ = st.columns([1,1,5])
    with c1: start_btn = st.button("▶️ Démarrer",type="primary",use_container_width=True)
    with c2: stop_btn  = st.button("⏹️ Arrêter", use_container_width=True)
 
    if start_btn: st.session_state['running']=True
    if stop_btn:  st.session_state['running']=False
 
    if st.session_state.get('running',False):
        buffer=collections.deque(maxlen=128)
        hist_ax,hist_ay,hist_az=[],[],[]
        n_fog=n_normal=t=0
        last_pred=0; last_proba=0.0
 
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
 
            if last_pred==1:
                diag_box.markdown('<div class="diag-box fog"><div class="diag-icon">🔴</div><div class="diag-label">FoG DÉTECTÉ !</div><div class="diag-sub">Enrayement cinétique</div></div>', unsafe_allow_html=True)
            else:
                diag_box.markdown('<div class="diag-box normal"><div class="diag-icon">🟢</div><div class="diag-label">Marche Normale</div><div class="diag-sub">Aucune anomalie</div></div>', unsafe_allow_html=True)
 
            bc="#E74C3C" if last_proba>60 else "#27AE60"
            proba_box.markdown(f'<div class="proba-container"><div class="proba-label">Probabilité FoG</div><div class="proba-bar-bg"><div class="proba-bar-fill" style="width:{last_proba}%;background:{bc};"></div></div><div class="proba-value" style="color:{bc};">{last_proba}%</div></div>', unsafe_allow_html=True)
            stats_box.markdown(f'<div style="background:white;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.05);"><div style="font-size:0.7rem;font-weight:600;color:#7F8C8D;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Stats Session</div><div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="color:#27AE60;font-weight:600;">🟢 Normal</span><span style="font-family:Playfair Display;font-weight:700;color:#0A2342;">{n_normal}</span></div><div style="display:flex;justify-content:space-between;"><span style="color:#E74C3C;font-weight:600;">🔴 FoG</span><span style="font-family:Playfair Display;font-weight:700;color:#E74C3C;">{n_fog}</span></div></div>', unsafe_allow_html=True)
 
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
    else:
        with col_chart:
            st.markdown('<div style="background:white;border-radius:14px;padding:48px;text-align:center;box-shadow:0 2px 12px rgba(0,0,0,0.06);"><div style="font-size:3rem;margin-bottom:12px;">▶️</div><div style="font-family:Playfair Display,serif;font-size:1.2rem;font-weight:600;color:#0A2342;">Cliquez sur Démarrer pour lancer la simulation</div></div>', unsafe_allow_html=True)
 
