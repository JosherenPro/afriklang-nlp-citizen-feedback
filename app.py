"""Démo : classification de commentaires citoyens (FR + éwé) — streamlit run app.py"""
import html
import re
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from preprocessing import EWE_LOOKUP, glossary_key

MODELS = Path(__file__).parent / "models"
DEMO_CSV = Path(__file__).parent / "test_mixte_fr_ewe.csv"
COLORS = {"Satisfaction": "#1baf7a", "Insatisfaction": "#eb6834", "Suggestion": "#2a78d6"}
BADGES = {"Satisfaction": ("green", ":material/sentiment_satisfied:"),
          "Insatisfaction": ("orange", ":material/sentiment_dissatisfied:"),
          "Suggestion": ("blue", ":material/lightbulb:")}
EXAMPLES = ["Mele via o, service la mègbe trop!", "Akpe na wò, service très professionnel",
            "Le personnel a été accueillant et efficace.", "Trois semaines d'attente pour un simple certificat.",
            "Créer une application mobile pour les démarches"]
REPO = "https://github.com/JosherenPro/afriklang-nlp-citizen-feedback"
MAX_WORDS = 80  # ponytail: occlusion = 1 encodage par mot ; au-delà, on tronque l'explication


@st.cache_resource
def load_models():
    from sentence_transformers import SentenceTransformer
    e5 = joblib.load(MODELS / "e5_logreg.joblib")
    e5["model"] = SentenceTransformer(e5["encoder"], device="cpu")
    return e5, joblib.load(MODELS / "tfidf_logreg.joblib")


def predict_proba(texts, model_name):
    """Probabilités (n_textes × 3). e5 reçoit le texte brut, comme à l'entraînement (notebook, partie 2)."""
    e5, tfidf = load_models()
    if model_name == "TF-IDF":
        return tfidf.predict_proba(texts)
    embeddings = e5["model"].encode([e5["prefix"] + t for t in texts], normalize_embeddings=True, batch_size=32)
    return e5["classifier"].predict_proba(embeddings)


def classes(model_name):
    e5, tfidf = load_models()
    return list((tfidf if model_name == "TF-IDF" else e5["classifier"]).classes_)


def analyse(text, model_name):
    """Classe prédite, probabilités, et contribution de chaque mot par occlusion (un seul batch)."""
    words = text.split()[:MAX_WORDS]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = predict_proba([text] + variants, model_name)
    best = probs[0].argmax()
    return classes(model_name)[best], probs[0], list(zip(words, probs[0, best] - probs[1:, best]))


def ewe_words(text):
    return {w: EWE_LOOKUP[glossary_key(w)] for w in re.findall(r"[\ẁ-ͯ]+", text.lower())
            if glossary_key(w) in EWE_LOOKUP}


def highlight(contributions, color):
    top = max([abs(c) for _, c in contributions] + [1e-9])
    spans = []
    for word, c in contributions:
        alpha = int(min(abs(c) / top, 1) * 200)
        bg = f"{color}{alpha:02x}" if c > 0 else f"#888888{alpha // 2:02x}"
        spans.append(f'<span title="{c:+.3f}" style="background:{bg};padding:2px 4px;border-radius:4px">'
                     f"{html.escape(word)}</span>")
    return f'<p style="line-height:2.2;font-size:1.1rem">{" ".join(spans)}</p>'


st.set_page_config(page_title="Commentaires citoyens", page_icon=":material/forum:")
st.title("Commentaires citoyens", icon=":material/forum:")
st.markdown("Classe un retour sur un service public en :green-badge[Satisfaction] :orange-badge[Insatisfaction] "
            ":blue-badge[Suggestion] — en français, avec ou sans mots d'éwé/mina — et montre les mots qui ont décidé.")

with st.sidebar:
    model_name = st.radio("Modèle", ["e5-base", "TF-IDF"],
                          captions=["retenu (F1 CV 0,981)", "repli explicable (F1 CV 0,717)"])
    st.subheader("À propos", icon=":material/info:")
    st.caption("Modèle : encodeur multilingual-e5-base figé + régression logistique (F1 macro en CV : 0,981).")
    st.caption("Limites : 150 textes d'entraînement seulement ; l'éwé n'est couvert que par un petit glossaire.")
    st.caption("Explication : baisse de probabilité de la classe prédite quand on retire chaque mot.")
    st.link_button("Notebook et code", REPO, icon=":material/code:", width="stretch")

single_tab, batch_tab = st.tabs(
    [":material/chat: Un commentaire", ":material/table_chart: Un fichier CSV"])

with single_tab:
    st.pills("Exemples", EXAMPLES, key="example",
             on_change=lambda: st.session_state.update(text=st.session_state.example or st.session_state.text))
    text = st.text_area("Commentaire", key="text", placeholder="Écrivez un commentaire…", height=100)
    st.button("Analyser", type="primary", icon=":material/search:")

    if text.strip():
        with st.spinner("Analyse… (le premier appel charge le modèle, ~20 s)"):
            label, probs, contributions = analyse(text, model_name)

        with st.container(border=True):
            color, icon = BADGES[label]
            verdict, chart = st.columns([2, 3], vertical_alignment="center")
            verdict.caption("Classe prédite")
            verdict.badge(label, icon=icon, color=color)
            # Pas de « % de confiance » : les probabilités de la régression logistique sur e5 ne sont pas calibrées
            # (médiane 0,56 même sur les textes d'entraînement) ; seul leur classement est fiable.
            df = pd.DataFrame({"classe": classes(model_name), "probabilité": probs})
            chart.altair_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("probabilité", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("classe", title=None, sort=list(COLORS), axis=alt.Axis(labelOverlap=False)),
                color=alt.Color("classe", scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values())), legend=None),
                tooltip=["classe", alt.Tooltip("probabilité", format=".1%")]).properties(height=150))

        st.subheader("Mots qui ont pesé", icon=":material/highlight:")
        st.html(highlight(contributions, COLORS[label]))
        st.caption(f"Couleur {label.lower()} : le mot pousse vers « {label} » (plus foncé = plus fort). "
                   "Gris : le mot tire vers une autre classe. Survolez un mot pour sa contribution.")

        if glosses := ewe_words(text):
            st.subheader("Mots éwé détectés", icon=":material/translate:")
            st.dataframe(pd.DataFrame({"éwé / mina": list(glosses), "glose française": list(glosses.values())}),
                         hide_index=True)
            if model_name == "e5-base":
                st.caption("e5 lit le texte brut (comme à l'entraînement) ; le glossaire sert au TF-IDF et à cet affichage.")

with batch_tab:
    # Cas d'usage réel d'une administration : classer d'un coup un export de retours citoyens.
    uploaded = st.file_uploader("Fichier CSV (une colonne de commentaires)", type="csv")
    use_demo = st.toggle("Essayer avec le jeu de test mixte français/éwé (15 textes relus par un locuteur)",
                         value=uploaded is None)
    if uploaded is not None or use_demo:
        comments = pd.read_csv(uploaded if uploaded is not None else DEMO_CSV)
        text_columns = list(comments.select_dtypes(include=["object", "str"]).columns)
        if not text_columns:
            st.error("Aucune colonne de texte dans ce fichier.", icon=":material/error:")
        else:
            column = st.selectbox("Colonne à classer", text_columns,
                                  index=text_columns.index("texte") if "texte" in text_columns else 0)
            texts = comments[column].fillna("").astype(str)
            with st.spinner(f"Classement de {len(texts)} commentaires…"):
                batch_probs = predict_proba(list(texts), model_name)
            labels = classes(model_name)
            # Classe prédite juste après le texte, pour qu'elle reste visible sans défilement horizontal
            comments.insert(comments.columns.get_loc(column) + 1, "classe prédite",
                            [labels[i] for i in batch_probs.argmax(axis=1)])
            counts = comments["classe prédite"].value_counts().reindex(list(COLORS), fill_value=0)
            # Une colonne dont toutes les valeurs sont des classes connues sert de vérité terrain : on évalue.
            label_columns = [c for c in comments.columns if c != "classe prédite"
                             and comments[c].dropna().isin(list(COLORS)).all() and comments[c].notna().any()]
            metric_columns = st.columns(4 if label_columns else 3, border=True)
            for metric_column, (name, count) in zip(metric_columns, counts.items()):
                metric_column.metric(name, f"{count} · {count / len(comments):.0%}")
            if label_columns:
                correct = comments["classe prédite"] == comments[label_columns[0]]
                comments.insert(comments.columns.get_loc("classe prédite") + 1, "juste", correct)
                metric_columns[3].metric("Justes", f"{correct.sum()}/{len(correct)}",
                                         help=f"Comparaison avec la colonne « {label_columns[0]} » du fichier.")
            st.dataframe(comments, hide_index=True,
                         column_config={"juste": st.column_config.CheckboxColumn("juste", width="small")})
            st.download_button("Télécharger le CSV classé", comments.to_csv(index=False).encode("utf-8"),
                               "commentaires_classes.csv", "text/csv", icon=":material/download:", type="primary")
