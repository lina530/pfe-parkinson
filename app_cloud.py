# ============================================
# app_cloud.py — Application Streamlit Cloud
# Tout en un : traitement + ML + dashboard
# ============================================
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
import joblib
import collections
import time
import os

st.set_page_config(
    page_title="Détection Parkinson — PFE",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# CHARGEMENT DU MODÈLE
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
# FONCTIONS DE TRAITEMENT
# ============================================
def butterworth_filter(signal, cutoff=3, fs=64, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, signal)

def extract_features(window, fs=64):
    features = []
    for axis in range(window.shape[1]):
        sig = window[:, axis]
        features.append(np.mean(sig))
        features.append(np.std(sig))
        features.append(np.max(sig))
        features.append(np.min(sig))
        features.append(np.max(sig) - np.min(sig))
        features.append(np.sqrt(np.mean(sig**2)))
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs    = np.fft.rfftfreq(len(sig), d=1/fs)
        features.append(np.sum(fft_vals))
        freeze  = fft_vals[(freqs>=3)  & (freqs<=8)]
        locomot = fft_vals[(freqs>=0.5)& (freqs<=3)]
        fi = np.sum(freeze**2) / (np.sum(locomot**2) + 1e-6)
        features.append(fi)
    return np.array(features)

def predict_window(window, model, scaler):
    w = window.copy()
    for i in range(3):
        w[:, i] = butterworth_filter(w[:, i])
    feat   = extract_features(w).reshape(1, -1)
    feat_s = scaler.transform(feat)
    pred   = model.predict(feat_s)[0]
    proba  = model.predict_proba(feat_s)[0][1]
    return int(pred), round(float(proba)*100, 1)

# ============================================
# INTERFACE
# ============================================

# ---- HEADER ----
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🧠")
with col_title:
    st.title("Système de Détection de la Maladie de Parkinson")
    st.caption("Analyse de Marche par Capteurs IMU & IA — PFE Ingénierie Biomédicale")

st.divider()

# ---- SIDEBAR ----
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Plus_symbol.svg/100px-Plus_symbol.svg.png", width=50)
    st.header("⚙️ Configuration")
    st.divider()

    mode = st.radio(
        "Mode d'utilisation",
        ["📁 Fichier CSV/TXT", "🎲 Simulation Demo"],
        index=1
    )
    st.divider()

    st.subheader("👤 Patient")
    nom_patient = st.text_input("Nom", "Patient 001")
    age_patient = st.number_input("Âge", min_value=0, max_value=120, value=65)
    st.divider()

    st.subheader("🔬 À propos")
    st.info(
        "**PFE — Ingénierie Biomédicale**\n\n"
        "Détection automatique du Freezing of Gait (FoG) "
        "par capteurs IMU et Machine Learning.\n\n"
        "Encadré par : Pr. Youssef Ibnatta"
    )

# ============================================
# MODE 1 : UPLOAD FICHIER
# ============================================
if mode == "📁 Fichier CSV/TXT":

    st.subheader("📂 Charger un fichier de données IMU")

    uploaded = st.file_uploader(
        "Glisse un fichier .txt (format Daphnet) ou .csv",
        type=["txt", "csv"]
    )

    if uploaded is not None:
        col_names = ['time',
                     'ankle_x','ankle_y','ankle_z',
                     'thigh_x','thigh_y','thigh_z',
                     'trunk_x','trunk_y','trunk_z',
                     'label']
        try:
            df = pd.read_csv(uploaded, sep=" ", header=None, names=col_names)
            df = df[df['label'] != 0].reset_index(drop=True)

            st.success(f"✅ Fichier chargé : {len(df)} échantillons")

            # Analyse complète
            WINDOW_SIZE = 128
            OVERLAP     = 64

            windows, labels_pred, labels_real = [], [], []

            for start in range(0, len(df) - WINDOW_SIZE, OVERLAP):
                end    = start + WINDOW_SIZE
                window = df[['ankle_x','ankle_y','ankle_z']].iloc[start:end].values
                label  = df['label'].iloc[start:end].mode()[0]

                if model_ok:
                    pred, proba = predict_window(window, model, scaler)
                    labels_pred.append(pred)
                labels_real.append(1 if label == 2 else 0)
                windows.append(start/64)

            # ---- Résultats ----
            col1, col2, col3 = st.columns(3)

            fog_detected = sum(labels_pred) if model_ok else 0
            fog_real     = sum(labels_real)

            with col1:
                st.metric("⏱️ Durée totale",
                          f"{len(df)/64:.0f} secondes")
            with col2:
                st.metric("🔴 Épisodes FoG réels",
                          f"{fog_real} fenêtres")
            with col3:
                if model_ok:
                    st.metric("🤖 FoG détectés par IA",
                              f"{fog_detected} fenêtres")

            # ---- Graphique signal ----
            st.subheader("📈 Signal Accéléromètre")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df['ankle_z'].values[:2000],
                mode='lines', name='Axe Z',
                line=dict(color='#0077b6', width=1)))

            # Zones FoG en rouge
            fog_zones = df[df['label']==2].index[:2000]
            if len(fog_zones) > 0:
                fig.add_vrect(
                    x0=fog_zones[0], x1=fog_zones[-1],
                    fillcolor="red", opacity=0.15,
                    annotation_text="Zone FoG",
                    annotation_position="top left"
                )

            fig.update_layout(
                height=300,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis_title="Échantillons",
                yaxis_title="Accélération (mg)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---- Prédictions dans le temps ----
            if model_ok and len(labels_pred) > 0:
                st.subheader("🤖 Prédictions IA dans le temps")
                fig2 = go.Figure()
                colors = ['red' if p==1 else 'green' for p in labels_pred]
                fig2.add_trace(go.Bar(
                    x=windows,
                    y=labels_pred,
                    marker_color=colors,
                    name='Prédiction (1=FoG)'
                ))
                fig2.update_layout(
                    height=200,
                    plot_bgcolor='#0e1117',
                    paper_bgcolor='#0e1117',
                    font=dict(color='white'),
                    xaxis_title="Temps (secondes)",
                    yaxis_title="FoG Détecté"
                )
                st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {e}")

    else:
        st.info("👆 Charge un fichier .txt du dataset Daphnet pour commencer l'analyse")

# ============================================
# MODE 2 : SIMULATION DEMO
# ============================================
else:
    st.subheader("🎲 Mode Simulation — Démonstration en Temps Réel")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("🔬 Diagnostic")
        pred_box  = st.empty()
        proba_box = st.empty()
        fi_box    = st.empty()
        st.divider()
        st.subheader("📊 Stats Session")
        stats_box = st.empty()

    with col1:
        chart_box = st.empty()

    # Bouton Start/Stop
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        start_btn = st.button("▶️ Démarrer", type="primary")
    with col_btn2:
        stop_btn  = st.button("⏹️ Arrêter")

    if start_btn:
        st.session_state['running'] = True
    if stop_btn:
        st.session_state['running'] = False

    # Simulation
    if st.session_state.get('running', False):

        buffer   = collections.deque(maxlen=128)
        hist_az  = []
        hist_pred = []
        n_fog    = 0
        n_normal = 0
        t        = 0

        while st.session_state.get('running', False):

            # Génère données simulées
            is_fog = (t % 300 > 200)  # FoG toutes les 300 itérations

            if is_fog:
                # Signal FoG : haute variabilité + fréquence élevée
                ax = np.random.normal(300, 80)
                ay = np.random.normal(950, 120)
                az = np.random.normal(100, 200) + 150*np.sin(2*np.pi*5*t/64)
            else:
                # Marche normale : signal régulier
                ax = np.random.normal(50,  20)
                ay = np.random.normal(980, 30)
                az = np.random.normal(-80, 40) + 60*np.sin(2*np.pi*1.8*t/64)

            buffer.append([ax, ay, az])
            hist_az.append(az)
            if len(hist_az) > 256:
                hist_az = hist_az[-256:]

            # Prédiction quand buffer plein
            if len(buffer) == 128 and model_ok:
                window = np.array(buffer)
                pred, proba = predict_window(window, model, scaler)

                if pred == 1:
                    n_fog += 1
                    pred_box.error("## 🔴 FoG DÉTECTÉ !")
                else:
                    n_normal += 1
                    pred_box.success("## 🟢 Marche Normale")

                proba_box.metric(
                    "Probabilité FoG", f"{proba}%",
                    delta="⚠️ Risque élevé" if proba > 60 else "✅ Risque faible"
                )
                stats_box.info(
                    f"🔴 FoG : **{n_fog}** fenêtres\n\n"
                    f"🟢 Normal : **{n_normal}** fenêtres"
                )

            # Graphique
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=hist_az, mode='lines',
                name='Axe Z',
                line=dict(
                    color='red' if is_fog else '#0077b6',
                    width=1.5
                )
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=10,r=10,t=30,b=10),
                title=dict(
                    text="🔴 Signal FoG" if is_fog else "🟢 Marche Normale",
                    font=dict(color='red' if is_fog else 'lightgreen')
                ),
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117',
                font=dict(color='white'),
                xaxis_title="Échantillons",
                yaxis_title="Accélération (mg)",
                showlegend=False
            )
            chart_box.plotly_chart(fig, use_container_width=True)

            t += 1
            time.sleep(0.05)
    else:
        col1.info("👆 Clique sur **▶️ Démarrer** pour lancer la simulation")
        pred_box.info("## ⏳ En attente...")
