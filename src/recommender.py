"""Système de recommandation par similarité de descriptions.

Principe : pour un produit donné, on propose d'abord les produits du
même cluster, classés par similarité cosinus décroissante. En dernier
recours (outlier ou cluster trop petit), on complète avec les produits
les plus proches du catalogue.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_matrix(tfidf_matrix):
    """
    Calcule la similarité cosinus entre tous les produits.
    """
    return cosine_similarity(tfidf_matrix)


def find_similar_items(item_id, dataframe, similarity_matrix, n_items=5):
    """
    Recommande n produits proches du produit donné.
    """
    matches = dataframe.index[dataframe["id"] == item_id]
    if len(matches) == 0:
        raise ValueError(f"ID produit inconnu : {item_id}")

    position = dataframe.index.get_loc(matches[0])
    target_cluster = dataframe["cluster"].iloc[position]

    # Produits les plus proches d'abord, le produit lui-même exclu
    ranked = [
        i for i in np.argsort(similarity_matrix[position])[::-1]
        if i != position
    ]

    # Les outliers n'ont pas de voisins de cluster fiables
    if target_cluster == -1:
        selected = ranked[:n_items]
    else:
        same_cluster = dataframe["cluster"].to_numpy() == target_cluster
        in_cluster = [i for i in ranked if same_cluster[i]]
        others = [i for i in ranked if not same_cluster[i]]
        selected = (in_cluster + others)[:n_items]

    return dataframe["id"].iloc[selected].tolist()
