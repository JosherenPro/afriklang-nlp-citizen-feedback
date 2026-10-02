"""Prétraitement partagé par le notebook et l'application Streamlit."""
import re
import unicodedata

import nltk
from nltk.corpus import stopwords

# Gloses du corpus (ids 73, 131, 139) validées par un locuteur éwé ; les autres restent
# approximatives. Orthographe variable
# (avec ou sans tons, ɔ/o, ɛ/e…) : la recherche passe par glossary_key.
# Le corpus n'en contient que 3 commentaires (ids 73, 131, 139) ; le reste anticipe
# des retours citoyens futurs.
EWE_GLOSSARY = {
    # --- tokens présents dans le corpus ---
    "akpe": "merci",                 # id 139 ; validé : « Akpe na wò » = merci à toi
    "yevu": "européen étranger",     # id 73 ; validé : Blanc, Européen, étranger à peau claire (mina : yovo)
    "nyuie": "bien",                 # id 73
    "hafi": "vraiment",              # id 73 ; validé : « avant », mais en fin de phrase = insistance (« vraiment »)
    "mele": "ne pas",                # id 131 ; validé : me (je) + le (être) … o = « je ne suis/fais pas »
    "via": "demander",               # id 131 ; validé (aussi bia/biam) : « mele via o » = « je ne plaisante pas »
    "bia": "demander",
    "biam": "demander",
    "megbe": "retard lent",          # id 131 ; validé : derrière, en retard → service à la traîne
    # --- expressions courantes susceptibles d'apparaître ---
    "yovo": "européen étranger",
    "nyui": "bon",
    "nyo": "bon",                    # mina « é nyo » = c'est bon
    "enyo": "bon",
    "ŋutɔ": "très",                  # vraiment, très
    "vɔ̃": "mauvais mal",
    "gble": "gâté abîmé panne",      # « é gble » = c'est gâté
    "fũu": "beaucoup",
    "geɖe": "beaucoup",
    "sugbɔ": "beaucoup",
    "kakaka": "beaucoup",            # « akpe kakaka » = merci beaucoup
    "blewu": "lent lentement",
    "kaba": "vite rapide",
    "ga": "argent",
    "lalã": "attendre attente",
    "ayekoo": "bravo félicitations",
    "woezɔ": "bienvenue",
    "dzidzɔ": "joie content",
    "vivi": "agréable",
    "nublanui": "dommage triste",
    "dziku": "colère",
    "dɔwɔla": "agent travailleur",
    "agbalẽ": "document papier",
    "kpekpeɖeŋu": "aide",
    "gbe": "langue",
}

# Lettres éwé ramenées à leur voisine latine, pour la recherche dans le glossaire
# uniquement : sur clavier français on écrit souvent « vo » pour « vɔ̃ », « gede » pour « geɖe ».
_EWE_TO_LATIN = str.maketrans("ɔɛɖŋƒʋ", "oednfv")

# Élisions restituées en forme pleine : « n'ai » → « ne ai », pour que la négation
# élidée et la négation pleine comptent comme le même mot « ne ».
_ELISIONS = {"l": "le", "j": "je", "n": "ne", "d": "de", "qu": "que", "c": "ce", "s": "se", "m": "me", "t": "te"}
_ELISION_RE = re.compile(r"\b(qu|[ljndcsmt])['’]")

# \w Unicode garde é, ç, ɔ, ɖ, ŋ… ; les tons combinants (U+0300–036F, ex. le tilde
# de vɔ̃ qui n'a pas de forme précomposée) ne sont pas \w : on les autorise explicitement.
_NON_LETTER_RE = re.compile(r"[^\w\s̀-ͯ]|[\d_]")


def glossary_key(token):
    """Forme de recherche : sans accents ni tons, lettres éwé latinisées (vɔ̃ → vo)."""
    stripped = "".join(c for c in unicodedata.normalize("NFD", token) if not unicodedata.combining(c))
    return stripped.translate(_EWE_TO_LATIN)


EWE_LOOKUP = {glossary_key(key): gloss for key, gloss in EWE_GLOSSARY.items()}


def clean_text(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = _ELISION_RE.sub(lambda match: _ELISIONS[match.group(1)] + " ", text)
    text = _NON_LETTER_RE.sub(" ", text)
    # Glossaire additif : le token éwé reste, sa glose française est ajoutée derrière.
    words = [f"{word} {EWE_LOOKUP[glossary_key(word)]}" if glossary_key(word) in EWE_LOOKUP else word
             for word in text.split()]
    return " ".join(words)


try:
    NLTK_STOPWORDS_FR = set(stopwords.words("french"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    NLTK_STOPWORDS_FR = set(stopwords.words("french"))
# Retirés de la liste NLTK : la négation signe l'Insatisfaction, le conditionnel la
# Suggestion (« Il serait utile de… ») — les chiffres par classe sont dans le notebook.
# Les intensifs (très, bien, trop) et sans/aucun/jamais ne sont pas dans NLTK.
KEPT_FROM_NLTK = {"ne", "pas", "n",
                  "serais", "serait", "serions", "seriez", "seraient",
                  "aurais", "aurait", "aurions", "auriez", "auraient"}
STOPWORDS_FR = NLTK_STOPWORDS_FR - KEPT_FROM_NLTK


def tokenize(text, stop_words=STOPWORDS_FR):
    return [word for word in clean_text(text).split() if word not in stop_words and len(word) > 1]


if __name__ == "__main__":
    assert clean_text("Mele via o, service la mègbe trop!") == "mele ne pas via demander o service la mègbe retard lent trop"
    assert "akpe merci" in clean_text("Akpe na wò")
    assert clean_text("Nyuiɛ ɖe vɔ̃ ŋutɔ") == "nyuiɛ bien ɖe vɔ̃ mauvais mal ŋutɔ très"  # lettres et tons conservés
    assert clean_text("Je n'ai reçu qu’un accusé, 3 fois.") == "je ne ai reçu que un accusé fois"
    assert tokenize("Il serait bien de ne pas attendre.") == ["serait", "bien", "ne", "pas", "attendre"]
    assert "très" in tokenize("Très satisfait")
    print("preprocessing.py : tous les tests passent")
