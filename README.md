# Projet The North Face — Recommandation produit et structuration du catalogue

Certification CDSD RNCP35288 - Bloc 3 : Analyse prédictive de données structurées par l'intelligence artificielle

## Contexte

Projet de data science (NLP non supervisé) réalisé sur un corpus de **500 fiches produit** du catalogue The North Face.  
L'objectif métier : augmenter les ventes en ligne avec une section « Vous aimerez aussi » et une navigation par catégories mieux adaptée au catalogue.

## Objectifs

| # | Livrable | Technique |
|---|----------|-----------|
| 1 | Identifier des groupes de produits aux descriptions proches | TF-IDF + clustering **DBSCAN** (distance cosinus) |
| 2 | Recommander des produits similaires à un produit consulté | Clusters + **similarité cosinus** |
| 3 | Extraire les thèmes latents des descriptions | **LSA** (`TruncatedSVD`) |


## Données 

Jeu de données 'data/sample-data.csv (https://www.kaggle.com/cclark/product-item-data?select=sample-data.csv) (Kaggle) : colonnes `id` (entier de 1 à 500) et `description` (HTML).

## Etapes du projet

1. Imports et configuration
2. Exploration et préparation des données
3. Clustering DBSCAN
4. Système de recommandation
5. Topic modeling (LSA)


## Méthodologie

**Prétraitement**
Les 500 fiches contiennent du HTML et un gabarit commun (intitulés « Details », « Fabric », « Weight », pays de fabrication, phrase *Common Threads Recycling Program* présente dans 352 fiches). Ces éléments sont supprimés avant la lemmatisation avec spaCy, sinon ils dominent la représentation TF-IDF. Trois termes génériques sont ajoutés aux stop words (`fabric`, `provide`, `feature`), choisis d'après leur fréquence documentaire.

**Clustering**
DBSCAN avec distance cosinus. Un `eps` trop grand fait fusionner presque tout le catalogue dans un seul cluster ; un `eps` trop petit classe l'essentiel en bruit (`eps = 0,2` : 90 % de bruit). Une **grille de 60 couples** (`eps` × `min_samples`) est évaluée selon le nombre de clusters, le taux de bruit, la taille du plus gros cluster et la silhouette. Critères de sélection : 10 à 20 clusters, au plus 25 % de bruit, aucun cluster au-dessus de 30 % du catalogue, puis meilleure silhouette. Un seul couple les respecte (`eps = 0,68`, `min_samples = 5`).

**Recommandeur**
`find_similar_items(item_id, dataframe, similarity_matrix)` renvoie la liste des 5 `id` les plus proches : d'abord les produits du même cluster, triés par similarité cosinus ; si le produit est un outlier ou si son cluster est trop petit, la liste est complétée par les produits les plus proches du catalogue.

**Topic modeling**
`TruncatedSVD` sur la matrice TF-IDF (15 composantes). Un `argmax` sur les valeurs brutes place 235 produits sur 500 dans le topic 1 (vocabulaire « moyen » du catalogue). Les topics sont donc **standardisés (z-score)** avant de retenir le topic principal de chaque produit : le topic 1 retombe à 32 produits.



## Résultats

| Indicateur | Valeur |
|------------|--------|
| Vocabulaire TF-IDF | 2 073 termes (500 documents) |
| Clusters DBSCAN (`eps = 0,68`, `min_samples = 5`) | **20 clusters**, 23,4 % de bruit |
| Plus gros cluster | 25,4 % du catalogue (contre 59 % avec un premier réglage) |
| Silhouette (hors bruit) | 0,215, soit une structure faible : les clusters se chevauchent |
| Qualité des recommandations | similarité moyenne **0,477** contre 0,070 au hasard (×6,8) |
| Topics LSA | 15 topics, 25,7 % de variance expliquée |
| Cohérence clusters / topics | pureté de 71 % |

**Exemple de recommandation** — pour la *Down sweater vest* (ID 35) : *Down sweater vest* (ID 282), deux *Down sweater* (IDs 473 et 179) et deux *Micro puff vest* (IDs 204 et 416), tous du même cluster « Doudounes et vestes isolées ».

Les visuels (wordclouds des clusters et des topics, courbe k-distance, compromis bruit / cluster géant) sont dans [`assets/images/`](assets/images/).


## Conclusion et limites
1. **Le catalogue se structure en familles de produits nettes** (20 clusters, 77 % des produits classés), sauf un cluster "pot commun" (25 %).
2. **La recommandation est cohérente** (×6,8 vs hasard), à valider sur des usages réels.
3. **LSA complète DBSCAN** : thèmes pour 100 % du catalogue, et subdivision du cluster 0.
4. **Limites** : silhouette basse (0,215), un seul couple de paramètres valide, quasi-doublons dans les recommandations, topics 12 à 15 peu lisibles, 500 fiches seulement.


## Pistes d'amélioration

- Comparer avec HDBSCAN ou k-means, ou clusteriser sur la matrice réduite par SVD.
- Utiliser des embeddings de phrases à la place du TF-IDF.
- Dédoublonner les variantes d'un même produit dans les recommandations.
- Mesurer l'effet réel du recommandeur par un test A/B (taux de clic, conversion).


## Structure du repo

```
cdsd_bloc3_projet_the_north_face /
├── assets
│   ├── images
│   │   ├── 01_tfidf_top20.png
│   │   ├── 02_eps_tradeoff.png
│   │   ├── 03_k_distance.png
│   │   ├── 04_wordclouds_clusters.png
│   │   └── 05_wordclouds_topics.png
│   └── dbscan_grid.csv
├── data
│   └── sample-data.csv
├── notebook
│   └── projet_the_north_face.ipynb
├── src
│   ├── preprocessing.py
│   └── recommender.py
├── .gitignoreb
├── README.md
└── requirements.txt
```

## Installation et exécution

```powershell
conda create -n cdsd_b3_projet_the_north_face python=3.13.14 pip -y
conda activate cdsd_b3_projet_the_north_facer
python -m pip install -r requirements.txt    
```

- La dernière cellule de la partie 2 appelle `input()` : saisir l'`id` d'un produit (1 à 500) pour obtenir 5 recommandations.





