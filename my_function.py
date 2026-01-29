# -*- coding: utf-8 -*-
"""
Script de classification forestière par Random Forest.
Optimisé pour les données Sentinel-2 et les indices de végétation.
"""
# Bibliothèques standards
import os
import warnings

# scientifiques / données
import numpy as np
import pandas as pd
import geopandas as gpd

# Graphiques
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

# SIG
from osgeo import gdal, ogr

#Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# Bibliothèques personnels
from libsigma import plots as pl
from libsigma import image_visu as iv


# On ignore les warnings liés aux divisions par zéro dans les calculs d'indices
warnings.filterwarnings('ignore', category=RuntimeWarning)


# Création de l'arborescence demandée
def creation_dossier():
    """Crée l'arborescence de sortie pour les résultats à l'extérieur du dépôt."""
    # On définit les chemins en remontant d'un niveau (..)
    dossiers = [
        os.path.join("..", "results"),
        os.path.join("..", "results", "figure")
    ]
    
    for dossier in dossiers:
        os.makedirs(dossier, exist_ok=True)
        print(f"Dossier vérifié/créé : {os.path.abspath(dossier)}")


# Préparation et rasterisation du fichier de polygones échantillons (interprétation satellite)
def rasterize_shapefile(shp_path, ref_raster_path, out_raster, field="strate"):
    """
    Rasterise un fichier vecteur (polygones) en une image raster alignée
    spatialement sur un raster de référence ici b3.

    Chaque pixel du raster de sortie prend la valeur de l'attribut spécifié
    dans le champ `field` du shapefile.

    Paramètres
    ----------
    shp_path : str
        Chemin vers le fichier shapefile contenant les polygones à rasteriser.
    ref_raster_path : str
        Chemin vers le raster de référence (résolution, emprise et projection).
    out_raster : str
        Chemin du fichier raster de sortie (format GeoTIFF).
    field : str, optionnel
        Nom du champ attributaire du shapefile à utiliser pour la rasterisation
        (par défaut : "strate").
    """

    # Ouverture du raster de référence afin de récupérer sa géométrie
    # (taille, résolution, projection et géoréférencement)
    ref_ds = gdal.Open(ref_raster_path)

    # Sélection du driver GeoTIFF pour la création du raster de sortie
    driver = gdal.GetDriverByName('GTiff')

    # Création du raster de sortie avec les mêmes dimensions que le raster de référence
    # Une seule bande est créée, de type entier (Int32)
    out_ds = driver.Create(
        out_raster,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        1,
        gdal.GDT_Int32
    )

    # Copie du géoréférencement et du système de projection du raster de référence
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())

    # Initialisation de la bande raster avec une valeur NoData
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.Fill(-9999)

    # Ouverture du shapefile contenant les polygones à rasteriser
    shp_ds = ogr.Open(shp_path)

    # Rasterisation des polygones :
    # - les valeurs du champ attributaire `field` sont affectées aux pixels
    # - l'option ALL_TOUCHED=TRUE permet d'affecter tous les pixels touchés
    #   par un polygone (et pas uniquement ceux dont le centre est inclus)
    gdal.RasterizeLayer(
        out_ds,
        [1],
        shp_ds.GetLayer(),
        options=[f"ATTRIBUTE={field}", "ALL_TOUCHED=TRUE"]
    )

    # Fermeture explicite du raster de sortie (écriture sur disque)
    out_ds = None



def analyser_donnees_entree(shp_path, strate_raster_path):
    """
    Analyse la distribution des échantillons d’apprentissage à partir :
    - des polygones du shapefile (vérité terrain),
    - des pixels rasterisés associés à chaque classe.

    La fonction génère et enregistre deux graphiques :
    1) la distribution du nombre de polygones par classe,
    2) la distribution du nombre de pixels par classe après rasterisation.
    
    Paramètres
    ----------
    shp_path : str
        Chemin vers le shapefile contenant les polygones d’échantillonnage.
    strate_raster_path : str
        Chemin vers le raster de classes issu de la rasterisation du shapefile.
    """

    # Configuration de la police par défaut pour assurer une homogénéité graphique
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    # Dictionnaire de correspondance entre les codes numériques et les noms de classes
    noms_classes = {
        1: 'Sol Nu',
        2: 'Herbe',
        3: 'Landes',
        4: 'Arbre'
    }

    # PRÉPARATION DES DONNÉES

    # Lecture du shapefile et comptage du nombre de polygones par classe
    gdf = gpd.read_file(shp_path)
    poly_counts = gdf['strate'].value_counts().sort_index()

    # Remplacement des codes numériques par les noms de classes
    poly_counts.index = [noms_classes[i] for i in poly_counts.index]

    # Lecture du raster de classes et comptage du nombre de pixels par classe
    ds = gdal.Open(strate_raster_path)
    arr = ds.ReadAsArray()

    # Sélection des pixels valides (> 0) et calcul des effectifs par classe
    classes, counts = np.unique(arr[arr > 0], return_counts=True)
    pixel_counts = pd.Series(
        counts,
        index=[noms_classes[c] for c in classes]
    )

    # GRAPHIQUE 1 : DISTRIBUTION DES POLYGONES

    # Création d'une figure dédiée pour les polygones
    plt.figure(figsize=(7, 6))
    ax1 = plt.gca()

    poly_counts.plot(
        kind='bar',
        color='skyblue',
        edgecolor='black',
        ax=ax1,
        zorder=3
    )

    # Ajout des valeurs numériques au-dessus des barres
    ax1.bar_label(ax1.containers[0], padding=3)

    # Mise en forme du graphique
    ax1.set_title("Distribution des Polygones", fontweight='bold')
    ax1.set_ylabel("Nombre de polygones")
    pl.custom_bg(ax1)
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Ajustement automatique et sauvegarde
    plt.tight_layout()
    plt.savefig('../results/figure/diag_baton_nb_poly_by_class.png')
    plt.show()

    # GRAPHIQUE 2 : DISTRIBUTION DES PIXELS

    # Création d'une nouvelle figure dédiée pour les pixels
    plt.figure(figsize=(7, 6))
    ax2 = plt.gca()

    pixel_counts.plot(
        kind='bar',
        color='salmon',
        edgecolor='black',
        ax=ax2,
        zorder=3
    )

    # Ajout des valeurs numériques au-dessus des barres
    ax2.bar_label(ax2.containers[0], padding=3)

    # Mise en forme du graphique
    ax2.set_title(
        "Distribution des Pixels (All Touched)",
        fontweight='bold'
    )
    ax2.set_ylabel("Nombre de pixels")
    pl.custom_bg(ax2)
    plt.setp(ax2.get_xticklabels(), rotation=0)

    # Ajustement automatique et sauvegarde
    plt.tight_layout()
    plt.savefig('../results/figure/diag_baton_nb_pix_by_class.png')
    plt.show()



def calcul_ari_serie(ds_b03, ds_b05):
    """
    Calcule l'indice ARI (Anthocyanin Reflectance Index) pour une
    série temporelle d'images Sentinel-2 empilées.

    L'indice ARI est calculé à partir des bandes B03 (vert) et B05
    (red-edge) pour chaque date de la série temporelle, selon la formule :
    
        ARI = (1 / B03 - 1 / B05) / (1 / B03 + 1 / B05)

    Les valeurs non définies (divisions par zéro, NaN, infinis) sont
    remplacées par une valeur NoData (-9999).

    Paramètres
    ----------
    ds_b03 : gdal.Dataset
        Dataset GDAL contenant la bande B03 empilée sur plusieurs dates.
    ds_b05 : gdal.Dataset
        Dataset GDAL contenant la bande B05 empilée sur plusieurs dates.

    Retour
    ------
    np.ndarray
        Tableau 3D de dimensions (lignes, colonnes, dates) contenant
        les valeurs de l'indice ARI pour chaque pixel et chaque date.
    """

    # Récupération des dimensions spatiales et temporelles
    nb_l = ds_b03.RasterYSize  # Nombre de lignes
    nb_c = ds_b03.RasterXSize  # Nombre de colonnes
    nb_d = ds_b03.RasterCount  # Nombre de dates (bandes empilées)

    # Initialisation du tableau de sortie (stack temporel)
    stack = np.zeros((nb_l, nb_c, nb_d), dtype=np.float32)

    # Boucle sur chaque date de la série temporelle
    for i in range(1, nb_d + 1):

        # Lecture des bandes B03 et B05 pour la date courante
        b03 = ds_b03.GetRasterBand(i).ReadAsArray().astype(np.float32)
        b05 = ds_b05.GetRasterBand(i).ReadAsArray().astype(np.float32)

        # Calcul de l'indice ARI en gérant les divisions par zéro
        with np.errstate(divide='ignore', invalid='ignore'):
            ari = (1.0 / b03 - 1.0 / b05) / (1.0 / b03 + 1.0 / b05)

            # Remplacement des valeurs infinies ou non définies par NoData
            ari[~np.isfinite(ari)] = -9999

        # Stockage de la couche ARI dans le stack temporel
        stack[:, :, i - 1] = ari

    return stack



def tracer_phenologie_ari(x_ari, y, dates_brutes):
    """
    Trace l'évolution temporelle de l'indice ARI pour chaque classe de
    couverture du sol à partir des pixels d'apprentissage.

    Pour chaque classe, la fonction calcule la moyenne et l'écart-type
    de l'indice ARI à chaque date, puis représente :
    - la signature phénologique moyenne,
    - une enveloppe de variabilité (± 1 écart-type).
    """

    # Conversion des dates au format datetime
    dates = pd.to_datetime(dates_brutes)

    # Création de la figure et de l'axe (syntaxe imposée)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    # Définition des couleurs et labels (ordre cohérent)
    colors = ['#A9A9A9', '#7CFC00', '#9370DB', '#006400']
    labels = ['Sol Nu', 'Herbe', 'Landes', 'Arbre']
    classes = [1, 2, 3, 4]

    # Boucle sur les classes avec zip (syntaxe identique à l'exemple)
    for val, color, label in zip(classes, colors, labels):

        # Sélection des pixels de la classe courante
        pixels = np.ma.masked_equal(x_ari[y == val], -9999)

        # Calcul des statistiques temporelles
        means = pixels.mean(axis=0)
        stds = pixels.std(axis=0)

        # Tracé de la moyenne
        ax.plot(dates, means, color=color, marker='o')

        # Tracé de l'enveloppe ± 1 écart-type
        ax.fill_between(
            dates,
            means - stds,
            means + stds,
            facecolor=color,
            alpha=0.3,
            label=label
        )

    # Mise en forme des axes
    ax.set_title("Signature phénologique de l'indice ARI", fontweight='bold')
    ax.legend()
    fig.autofmt_xdate()  # meilleure lisibilité des dates
    plt.tight_layout()

    # Sauvegarde et affichage
    plt.savefig('../results/figure/ARI_series.png', dpi=300)
    plt.show()



def entrainement_et_validation(x, y, groups, param_grid):
    """
    Optimise un modèle Random Forest et évalue sa robustesse spatiale 
    en utilisant une validation croisée groupée et reproductible.

    Paramètres
    ----------
    x : np.array
        Tableau des variables explicatives (bandes Sentinel-2 + indices dérivés).
    y : np.array
        Vecteur des classes cibles (étiquettes terrain).
    groups : np.array
        Identifiants des polygones pour la validation groupée (évite la fuite de données spatiales).
    param_grid : dict
        Dictionnaire des hyperparamètres à tester dans GridSearchCV.

    Retour
    -------
    tuple : (best_model, stats, best_params)
        best_model : RandomForestClassifier entraîné avec les meilleurs hyperparamètres.
        stats : dictionnaire contenant les performances de chaque itération :
                'acc' -> liste des précisions (accuracy),
                'cm' -> liste des matrices de confusion,
                'rep' -> liste des rapports de classification détaillés (DataFrame).
        best_params : dictionnaire des meilleurs hyperparamètres trouvés.
    """

    # --- 1. Recherche des meilleurs hyperparamètres avec GridSearchCV ---
    grid = GridSearchCV(
        RandomForestClassifier(random_state=0),  # RandomForest reproductible
        param_grid,                              # grille de paramètres à tester
        cv=GroupKFold(n_splits=5),             # validation croisée par groupes
        scoring='f1_weighted',       # critère d'optimisation : F1-score pondéré
        n_jobs=-1                                # parallélisation des tâches
    )
    # On entraîne le GridSearch sur toutes les données
    grid.fit(x, y, groups=groups)
    # On récupère le modèle entraîné avec les meilleurs paramètres
    best_mod = grid.best_estimator_

    # --- 2. Évaluation de la robustesse par validation croisée répétée ---
    stats = {'acc': [], 'rep': [], 'cm': []}  # dictionnaire pour stocker toutes les métriques
    nb_iter = 30                              # nombre de répétitions pour la robustesse
    nb_folds = 5                              # nombre de folds pour la validation croisée
    labels_classes = np.unique(y)             # liste des classes uniques

    for i in range(nb_iter):
        # Création des splits avec StratifiedGroupKFold pour garder la proportion des classes
        kf = StratifiedGroupKFold(
            n_splits=nb_folds, 
            shuffle=True, 
            random_state=0  # fixe le hasard pour rendre les splits reproductibles
        )
        
        for train_idx, test_idx in kf.split(x, y, groups=groups):
            # Création d'un nouveau modèle Random Forest pour chaque fold
            m = RandomForestClassifier(
                **grid.best_params_,  # on utilise les meilleurs hyperparamètres
                random_state=0,       # fixe le hasard pour reproductibilité
                n_jobs=-1
            )
            # Entraînement sur les données d'entraînement
            m.fit(x[train_idx], y[train_idx])
            # Prédiction sur les données de test
            p = m.predict(x[test_idx])
            
            # --- Stockage des métriques ---
            stats['acc'].append(accuracy_score(y[test_idx], p))  # précision globale
            stats['cm'].append(
                confusion_matrix(y[test_idx], p, labels=labels_classes))  # matrice de confusion
            
            # Rapport détaillé par classe sous forme de DataFrame
            report = classification_report(
                y[test_idx], p, output_dict=True, zero_division=0
            )
            df_rep = pd.DataFrame(report).iloc[:-1, :len(labels_classes)]
            stats['rep'].append(df_rep)
            
    # Retourne le meilleur modèle, les statistiques et les hyperparamètres optimaux
    return best_mod, stats, grid.best_params_


def afficher_bilan_performances(stats):
    """
    Génère et affiche un bilan global des performances du modèle de classification.

    La fonction produit :
    - une matrice de confusion moyenne calculée à partir de l'ensemble des
      validations croisées,
    - un graphique synthétique des performances par classe, intégrant la
      variabilité des résultats (qualité moyenne et dispersion).

    Paramètres
    ----------
    stats : dict
        Dictionnaire contenant les résultats des validations croisées :
        - 'cm'  : liste des matrices de confusion pour chaque itération,
        - 'rep' : liste des rapports de classification (DataFrame) par itération,
        - 'acc' : liste des précisions globales (accuracy).

    Retour
    ------
    None
        Génère et enregistre les graphiques de performance.
    """

    # --- 1. Calcul de la matrice de confusion moyenne ---
    # Moyenne des matrices de confusion obtenues sur toutes les itérations
    mean_cm = np.array(stats['cm']).mean(axis=0)

    # Noms des classes (ordre cohérent avec les étiquettes utilisées)
    labels_noms = ['Sol Nu', 'Herbe', 'Landes', 'Arbre']
    
    # Tracé et sauvegarde de la matrice de confusion moyenne
    pl.plot_cm(
        mean_cm,
        labels=labels_noms,
        out_filename='../results/figure/matrice_confusion_detaillee.png',
        normalize=True,     # affichage en pourcentage
        cmap='Greens'        # palette de couleurs
    )
    plt.show()

    # --- 2. Graphique de synthèse des performances par classe ---
    # Ce graphique combine :
    # - la qualité moyenne par classe (précision, rappel, F1-score),
    # - la précision globale (accuracy),
    # - la variabilité des performances entre itérations
    pl.plot_mean_class_quality(
        stats['rep'],
        stats['acc'],
        out_filename='../results/figure/performances_finales.png'
    )
    plt.show()



def afficher_importance(modele, noms_features):
    """
    Affiche les 20 variables les plus importantes du modèle Random Forest.

    L'importance des variables est calculée à partir de la diminution moyenne
    de l'impureté (Gini) fournie par le modèle Random Forest. Ce graphique
    permet d'identifier les bandes spectrales et indices les plus
    discriminants pour la classification.

    Paramètres
    ----------
    modele : RandomForestClassifier
        Modèle Random Forest entraîné.
    noms_features : list
        Liste des noms des variables explicatives (bandes et indices),
        dans le même ordre que celui utilisé pour l'entraînement.

    Retour
    ------
    None
        Génère et affiche un graphique des 20 variables les plus importantes.
    """

    # Création d'un DataFrame associant chaque variable à son importance
    imp = pd.DataFrame({
        'Feature': noms_features,
        'Imp': modele.feature_importances_
    })

    # Sélection des 20 variables les plus importantes
    top = imp.sort_values(by='Imp').tail(20)

    # Création du graphique horizontal
    plt.figure(figsize=(10, 8))
    plt.barh(top['Feature'], top['Imp'], color='teal')

    # Mise en forme du graphique
    plt.title("20 variables les plus explicatives", fontweight='bold')
    plt.xlabel("Importance relative")
    plt.tight_layout()

    # Affichage
    plt.show()



def afficher_importance_globale(modele, noms_features):
    """
    Affiche l'importance globale des variables explicatives en les regroupant
    par type (bandes spectrales et indices).

    Les importances individuelles des variables issues du modèle Random Forest
    sont agrégées par groupe de variables (ex. B02, NDVI, ARI), ce qui permet
    d'évaluer la contribution relative de chaque type de variable indépendamment
    de la dimension temporelle ou du nombre de répétitions.

    Paramètres
    ----------
    modele : RandomForestClassifier
        Modèle Random Forest entraîné.
    noms_features : list
        Liste des noms des variables explicatives, incluant les bandes
        spectrales et indices (ex. B02_20230715, NDVI_20230715).

    Retour
    ------
    None
        Génère et affiche un graphique de l'importance cumulée par groupe
        de variables.
    """

    # Création d'un DataFrame associant chaque variable à son importance
    imp = pd.DataFrame({
        'Feature': noms_features,
        'Imp': modele.feature_importances_
    })

    # Extraction du radical de la variable (ex. B02_date → B02)
    # afin de regrouper les variables par bande ou indice
    imp['Feature_Group'] = imp['Feature'].apply(
        lambda x: x.split('_')[0]
    )

    # Agrégation des importances par groupe de variables
    imp_glob = (
        imp.groupby('Feature_Group')['Imp']
        .sum()
        .reset_index()
        .sort_values(by='Imp', ascending=True)
    )

    # Création du graphique horizontal
    plt.figure(figsize=(10, 8))
    plt.barh(
        imp_glob['Feature_Group'],
        imp_glob['Imp'],
        color='teal'
    )

    # Mise en forme du graphique
    plt.xlabel("Importance cumulée")
    plt.title("Importance globale des variables", fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Affichage
    plt.show()



# --- 5. CARTOGRAPHIE ---

def generer_carte_finale(modele, liste_chemins, ari_path, out_path):
    """
    Applique le modèle de classification entraîné à l'ensemble des pixels
    d'une image Sentinel-2 afin de produire une carte finale d'occupation
    du sol.

    Les bandes spectrales, ainsi que l'indice ARI, sont empilées pour former
    un vecteur de variables explicatives par pixel. Le modèle Random Forest
    est ensuite utilisé pour prédire la classe associée à chaque pixel.

    Paramètres
    ----------
    modele : RandomForestClassifier
        Modèle Random Forest entraîné et optimisé.
    liste_chemins : list
        Liste des chemins vers les rasters des bandes spectrales Sentinel-2
        utilisées pour la classification.
    ari_path : str
        Chemin vers le raster de l'indice ARI.
    out_path : str
        Chemin de sortie pour la carte classifiée au format GeoTIFF.

    Retour
    ------
    None
        Génère et sauvegarde la carte de classification finale.
    """

    # Ouverture d'un raster de référence pour récupérer la géométrie
    ds_ref = gdal.Open(liste_chemins[0])

    # Création d'un masque pour exclure les pixels non valides
    # (zones sans données ou hors emprise)
    mask_data = np.all(ds_ref.ReadAsArray() > 0, axis=0)

    # Empilement des bandes et indices
    stack = []
    for chemin in liste_chemins + [ari_path]:
        data = gdal.Open(chemin).ReadAsArray()

        # Restructuration des données :
        # passage de (bandes, lignes, colonnes) à (n_pixels, n_features)
        stack.append(
            data.reshape(data.shape[0], -1).T
        )

    # Construction de la matrice finale de prédiction
    # Les valeurs NaN sont remplacées par 0 pour éviter les erreurs de prédiction
    x_img = np.nan_to_num(np.hstack(stack), nan=0.0)

    # Prédiction des classes pour l'ensemble des pixels
    pred = modele.predict(x_img).reshape(
        ds_ref.RasterYSize,
        ds_ref.RasterXSize
    )

    # Application du masque : les pixels invalides sont forcés à 0
    pred[mask_data == False] = 0

    # Création du raster de sortie GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        out_path,
        ds_ref.RasterXSize,
        ds_ref.RasterYSize,
        1,
        gdal.GDT_Byte
    )

    # Copie des informations géographiques
    out_ds.SetGeoTransform(ds_ref.GetGeoTransform())
    out_ds.SetProjection(ds_ref.GetProjection())

    # Écriture de la carte classifiée
    out_ds.GetRasterBand(1).WriteArray(pred)
    out_ds = None

    print(f"Carte sauvegardée : {out_path}")

def afficher_comparaison_finale(liste_chemins_rgb, chemin_carte):
    """
    Affiche côte à côte une image Sentinel-2 en vraies couleurs
    et la carte de classification issue du modèle Random Forest.

    La fonction assure une taille d'affichage identique pour les deux
    représentations afin de faciliter la comparaison visuelle entre
    l'image satellitaire et la carte thématique produite.

    Paramètres
    ----------
    liste_chemins_rgb : list
        Liste des chemins vers les bandes Sentinel-2 utilisées pour
        la composition RGB (une bande par élément).
    chemin_carte : str
        Chemin vers le raster de classification finale.

    Retour
    ------
    None
        Génère, sauvegarde et affiche une figure comparative.
    """

    # -----------------------------
    # Lecture et préparation de l'image Sentinel-2
    # -----------------------------
    layers = []

    for p in liste_chemins_rgb:
        # Ouverture du raster de la bande courante
        ds = gdal.Open(p)

        # Lecture de la bande (bande 8) et conversion en float
        bande = ds.GetRasterBand(8).ReadAsArray().astype(np.float32)

        # Masquage des valeurs invalides ou nulles
        bande[bande <= 0] = np.nan

        layers.append(bande)

    # Empilement des bandes pour former l'image RGB
    img_rgb = np.nan_to_num(np.dstack(layers))

    # -----------------------------
    # Lecture et préparation de la carte classée
    # -----------------------------
    ds_map = gdal.Open(chemin_carte)
    map_data = ds_map.ReadAsArray()

    # Masquage des pixels non classés (valeurs ≤ 0)
    map_masked = np.ma.masked_where(map_data <= 0, map_data)

    # -----------------------------
    # Création de la figure comparative
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # --- Affichage de l'image Sentinel-2 en vraies couleurs ---
    img_8bit = iv.rescale_to_8bits(
        img_rgb,
        stretch='percentile',
        percentiles=[2, 98]
    )
    ax1.imshow(img_8bit)
    ax1.set_title("Image Sentinel-2", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # --- Affichage de la carte de classification ---
    colors = ['#A9A9A9', '#7CFC00', '#9370DB', '#006400']
    custom_cmap = ListedColormap(colors)

    im = ax2.imshow(
        map_masked,
        cmap=custom_cmap,
        vmin=1,
        vmax=4,
        interpolation='none'
    )
    ax2.set_title(
        "Carte de Végétation (Random Forest)",
        fontsize=12,
        fontweight='bold'
    )
    ax2.axis('off')

    # Ajout de la légende colorimétrique sans modifier la taille des sous-figures
    cbar = fig.colorbar(im, ax=ax2, ticks=[1, 2, 3, 4], shrink=0.6)
    cbar.ax.set_yticklabels(['Sol Nu', 'Herbe', 'Landes', 'Arbre'])

    # -----------------------------
    # Finalisation, sauvegarde et affichage
    # -----------------------------
    plt.tight_layout()
    plt.savefig(
        '../results/figure/comparaison_sentinel_vs_classification.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()
