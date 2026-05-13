# ============================================
# RAPPORT PDF OPTIMISÉ (Correction encodage Unicode)
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
    # Remplacement du tiret long par un tiret standard ASCII "-"
    pdf.cell(0, 8, "NeuroGait - Évaluation Biomécanique", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(15, 20)
    pdf.cell(0, 6, "Analyse par capteurs inertiels (IMU) & IA - Rapport d'Examen")

    # Informations Patient
    pdf.set_xy(15, 42)
    pdf.set_text_color(13, 33, 55)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Dossier Patient", ln=True)
    pdf.set_draw_color(46, 134, 193)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    infos = [
        ("Identifiant", patient), 
        ("Âge / Sexe", f"{age} ans / {sexe}"), 
        ("Date d'analyse", date), 
        ("Opérateur", user)
    ]
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
    fill_c = (26, 86, 50) if score < 30 else ((120, 66, 18) if score < 65 else (160, 40, 30))
    pdf.set_fill_color(*fill_c)
    pdf.rect(15, pdf.get_y(), int(180 * score / 100), 8, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(15, pdf.get_y() + 0.5)
    pdf.cell(180, 7, f"   {score}/100 - Risque {niveau.upper()}", ln=True)

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

    # Remplacement des "—" par des "-" ou "N/A"
    rows = [
        ("Durée d'enregistrement", f"{duree:.1f} s", "-", "-"),
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
                if c in ["Anormal", "Critique", "Détecté"]: 
                    pdf.set_text_color(180, 40, 30)
                elif c == "Normal": 
                    pdf.set_text_color(26, 86, 50)
            pdf.cell(cw[j], 6, c, border=1, fill=True, align='C' if j > 0 else 'L')
        pdf.ln()

    # Pied de page
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 4, "NeuroGait v2.0 - Généré automatiquement. Validation clinique requise.", ln=True, align='C')
    return bytes(pdf.output())
