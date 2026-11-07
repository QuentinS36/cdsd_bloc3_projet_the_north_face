"""Prétraitement des descriptions produit The North Face.

Les fiches sont du HTML contenant un gabarit commun (intitulés,
poids, pays de fabrication...). Ce module nettoie ce bruit puis
lemmatise le texte avec spaCy.
"""
import re

import pandas as pd
import spacy

BOILERPLATE_PATTERNS = [
    r"<[^>]+>",                                    # balises HTML
    r"Recyclable through the Common Threads Recycling Program",
    r"\b(?:Details|Fabric|Weight):",               # intitulés du gabarit
    r"\(\s*\d+\s*g\s*[\d.]+\s*oz\s*\)",            # poids : (597 g 20.7 oz)
    r"Made in [A-Za-z ]+\.\s*$",                   # pays de fabrication
]

# Termes présents dans plus d'un tiers des fiches sans décrire le produit
CUSTOM_STOP_WORDS = ["fabric", "provide", "feature"]


def load_nlp(model="en_core_web_sm"):
    """
    Charge le modèle spaCy (sans parser ni NER, inutiles ici).
    """
    try:
        return spacy.load(model, disable=["parser", "ner"])
    except OSError:
        print(
            f"Modèle spaCy introuvable. "
        )
        raise


def remove_boilerplate(text):
    """
    Supprime les balises HTML et les phrases de gabarit
    communes à toutes les fiches produit.
    """
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text)
    return text


def lemmatize(doc):
    """Lemmatise un document spaCy, sans stopwords ni ponctuation."""
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
        and len(token) > 2
    ]
    return " ".join(tokens)


def clean_text(text, nlp):
    """
    Nettoie un texte brut :
        - suppression du HTML et du gabarit
        - passage en minuscule
        - lemmatisation
        - suppression des stopwords
        - suppression de la ponctuation
    """
    if pd.isna(text):
        return ""

    return lemmatize(nlp(remove_boilerplate(text)))


def clean_corpus(texts, nlp):
    """
    Nettoie tout un corpus (plus rapide qu'un appel clean_text par
    ligne : nlp.pipe traite les textes par lots).
    """
    cleaned = texts.fillna("").apply(remove_boilerplate)
    return [lemmatize(doc) for doc in nlp.pipe(cleaned)]
