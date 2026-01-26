# -*- coding: utf-8 -*-
"""
Script de classification forestière par Random Forest.
Optimisé pour les données Sentinel-2 et les indices de végétation.
"""

import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from osgeo import gdal, ogr

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, StratifiedGroupKFold
)
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)

# On ignore les warnings liés aux divisions par zéro dans les calculs d'indices
warnings.filterwarnings('ignore', category=RuntimeWarning)


# --- 1. ENVIRONNEMENT ---

import os

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


# --- 2. PRÉPARATION & RASTERISATION ---

def rasterize_shapefile(shp_path, ref_raster_path, out_raster, field="strate"):
    """
    Transforme un vecteur (polygones) en image raster alignée sur une référence.
    """
    ref_ds = gdal.Open(ref_raster_path)
    driver = gdal.GetDriverByName('GTiff')
    
    # Création du raster de sortie
    out_ds = driver.Create(
        out_raster, 
        ref_ds.RasterXSize, 
        ref_ds.RasterYSize, 
        1, 
        gdal.GDT_Int32
    )
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.Fill(-9999)
    
    # Rasterisation
    shp_ds = ogr.Open(shp_path)
    gdal.RasterizeLayer(
        out_ds, [1], 
        shp_ds.GetLayer(), 
        options=[f"ATTRIBUTE={field}", "ALL_TOUCHED=TRUE"]
    )
    out_ds = None


def analyser_donnees_entree(shp_path, strate_raster_path):
    """
    Affiche et enregistre séparément les statistiques des échantillons.
    """
    from libsigma import plots as pl
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import geopandas as gpd
    from osgeo import gdal
    import numpy as np
    import pandas as pd
    
    # Configuration police standard
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    noms_classes = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbre'}
    
    # --- PRÉPARATION DES DONNÉES ---
    # Analyse des polygones
    gdf = gpd.read_file(shp_path)
    poly_counts = gdf['strate'].value_counts().sort_index()
    poly_counts.index = [noms_classes[i] for i in poly_counts.index]

    # Analyse des pixels
    ds = gdal.Open(strate_raster_path)
    arr = ds.ReadAsArray()
    classes, counts = np.unique(arr[arr > 0], return_counts=True)
    pixel_counts = pd.Series(counts, index=[noms_classes[c] for c in classes])

    # --- GRAPHIQUE 1 : POLYGONES ---
    plt.figure(figsize=(7, 6)) # Création d'une figure dédiée
    ax1 = plt.gca()
    poly_counts.plot(
        kind='bar', color='skyblue', edgecolor='black', ax=ax1, zorder=3
    )
    ax1.bar_label(ax1.containers[0], padding=3)
    ax1.set_title("Distribution des Polygones", fontweight='bold')
    ax1.set_ylabel("Nombre de polygones")
    pl.custom_bg(ax1)
    plt.setp(ax1.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('../results/figure/diag_baton_nb_poly_by_class.png') # Sauvegarde unique 1
    plt.show()

    # --- GRAPHIQUE 2 : PIXELS ---
    plt.figure(figsize=(7, 6)) # Création d'une NOUVELLE figure dédiée
    ax2 = plt.gca()
    pixel_counts.plot(
        kind='bar', color='salmon', edgecolor='black', ax=ax2, zorder=3
    )
    ax2.bar_label(ax2.containers[0], padding=3)
    ax2.set_title("Distribution des Pixels (All Touched)", fontweight='bold')
    ax2.set_ylabel("Nombre de pixels")
    pl.custom_bg(ax2)
    plt.setp(ax2.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('../results/figure/diag_baton_nb_pix_by_class.png') # Sauvegarde unique 2
    plt.show()


# --- 3. CALCULS SPECTRAUX ---

def calcul_ari_serie(ds_b03, ds_b05):
    """Calcule l'indice ARI pour une série temporelle empilée."""
    nb_l = ds_b03.RasterYSize
    nb_c = ds_b03.RasterXSize
    nb_d = ds_b03.RasterCount
    
    stack = np.zeros((nb_l, nb_c, nb_d), dtype=np.float32)
    
    for i in range(1, nb_d + 1):
        b03 = ds_b03.GetRasterBand(i).ReadAsArray().astype(np.float32)
        b05 = ds_b05.GetRasterBand(i).ReadAsArray().astype(np.float32)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ari = (1.0/b03 - 1.0/b05) / (1.0/b03 + 1.0/b05)
            ari[~np.isfinite(ari)] = -9999
            
        stack[:, :, i-1] = ari
    return stack


def tracer_phenologie_ari(x_ari, y, dates_brutes):
    """Trace l'évolution temporelle de l'ARI par classe."""
    dates = pd.to_datetime(dates_brutes)
    plt.figure(figsize=(10, 6))
    
    colors = {1: '#A9A9A9', 2: '#7CFC00', 3: '#9370DB', 4: '#006400'}
    names = {1: 'Sol Nu', 2: 'Herbe', 3: 'Landes', 4: 'Arbre'}
    
    for val in [1, 2, 3, 4]:
        pixels = np.ma.masked_equal(x_ari[y == val], -9999)
        m, s = pixels.mean(axis=0), pixels.std(axis=0)
        plt.fill_between(dates, m-s, m+s, color=colors[val], alpha=0.15)
        plt.plot(dates, m, label=names[val], color=colors[val], marker='o')
        
    plt.title("Signature Phénologique ARI")
    plt.legend()
    plt.savefig('../results/figure/ARI_series.png')
    plt.show()


# --- 4. APPRENTISSAGE ET VALIDATION ---

def entrainement_et_validation(x, y, groups, param_grid):
    """
    Optimise le modèle Random Forest et valide sa robustesse spatiale.

    Paramètres:
    -----------
    x : np.array
        Tableau des variables explicatives (bandes Sentinel-2 + indices).
    y : np.array
        Vecteur des classes cibles (étiquettes terrain).
    groups : np.array
        Identifiants des polygones pour la validation groupée.
    param_grid : dict
        Dictionnaire des hyperparamètres pour le GridSearchCV.

    Return:
    ---------
    tuple : (best_model, stats, best_params)
    """
    # 1. Optimisation hyperparamètres
    grid = GridSearchCV(
        RandomForestClassifier(random_state=0), 
        param_grid, 
        cv=GroupKFold(n_splits=3), 
        scoring='accuracy', 
        n_jobs=-1
    )
    grid.fit(x, y, groups=groups)
    best_mod = grid.best_estimator_

    # 2. Évaluation robustesse
    stats = {'acc': [], 'rep': [], 'cm': []}
    nb_iter = 30
    nb_folds = 5
    labels_classes = np.unique(y)

    for i in range(nb_iter):
        kf = StratifiedGroupKFold(
            n_splits=nb_folds, 
            shuffle=True, 
            random_state=i
        )
        
        for train_idx, test_idx in kf.split(x, y, groups=groups):
            m = RandomForestClassifier(
                **grid.best_params_, 
                random_state=42, 
                n_jobs=-1
            )
            m.fit(x[train_idx], y[train_idx])
            p = m.predict(x[test_idx])
            
            # Stockage
            stats['acc'].append(accuracy_score(y[test_idx], p))
            stats['cm'].append(
                confusion_matrix(y[test_idx], p, labels=labels_classes)
            )
            
            report = classification_report(
                y[test_idx], p, output_dict=True, zero_division=0
            )
            df_rep = pd.DataFrame(report).iloc[:-1, :len(labels_classes)]
            stats['rep'].append(df_rep)
            
    return best_mod, stats, grid.best_params_


def afficher_bilan_performances(stats):
    """Génère les graphiques de bilan de performance (Matrice & Qualité)."""
    from libsigma import plots as pl
    
    # Matrice de confusion moyenne
    mean_cm = np.array(stats['cm']).mean(axis=0)
    labels_noms = ['Sol Nu', 'Herbe', 'Landes', 'Arbre']
    
    pl.plot_cm(
        mean_cm, 
        labels=labels_noms, 
        out_filename='../results/figure/matrice_confusion_detaillee.png',
        normalize=False, 
        cmap='Greens'
    )
    plt.show()

    # Barres d'erreurs
    pl.plot_mean_class_quality(
        stats['rep'], 
        stats['acc'], 
        out_filename='../results/figure/performances_finales.png'
    )
    plt.show()


def afficher_importance(modele, noms_features):
    """Affiche les 20 variables les plus importantes."""
    imp = pd.DataFrame({
        'Feature': noms_features, 
        'Imp': modele.feature_importances_
    })
    top = imp.sort_values(by='Imp').tail(20)
    
    plt.figure(figsize=(10, 8))
    plt.barh(top['Feature'], top['Imp'], color='teal')
    plt.title("20 variables les plus explicatives")
    plt.show()


def afficher_importance_globale(modele, noms_features):
    """Regroupe l'importance des variables par type (Bandes/Indices)."""
    imp = pd.DataFrame({
        'Feature': noms_features, 
        'Imp': modele.feature_importances_
    })
    
    # Extraction du radical (ex: B02_date -> B02)
    imp['Feature_Group'] = imp['Feature'].apply(lambda x: x.split('_')[0]) 
    
    imp_glob = imp.groupby('Feature_Group')['Imp'].sum().reset_index()
    imp_glob = imp_glob.sort_values(by='Imp', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(imp_glob['Feature_Group'], imp_glob['Imp'], color='teal')
    plt.xlabel("Importance cumulée")
    plt.title("Importance globale des variables")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# --- 5. CARTOGRAPHIE ---

def generer_carte_finale(modele, liste_chemins, ari_path, out_path):
    """Applique le modèle sur l'ensemble de l'image pour créer la carte."""
    ds_ref = gdal.Open(liste_chemins[0])
    mask_data = np.all(ds_ref.ReadAsArray() > 0, axis=0)
    
    stack = []
    for chemin in liste_chemins + [ari_path]:
        data = gdal.Open(chemin).ReadAsArray()
        # Reshape pour préparer le predict (n_pixels, n_features)
        stack.append(data.reshape(data.shape[0], -1).T)
    
    x_img = np.nan_to_num(np.hstack(stack), nan=0.0)
    pred = modele.predict(x_img).reshape(ds_ref.RasterYSize, ds_ref.RasterXSize)
    pred[mask_data == False] = 0
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        out_path, 
        ds_ref.RasterXSize, 
        ds_ref.RasterYSize, 
        1, 
        gdal.GDT_Byte
    )
    out_ds.SetGeoTransform(ds_ref.GetGeoTransform())
    out_ds.SetProjection(ds_ref.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(pred)
    out_ds = None
    print(f"Carte sauvegardée : {out_path}")


def afficher_comparaison_finale(liste_chemins_rgb, chemin_carte):
    """Affiche l'image Sentinel-2 et la carte classée côte à côte avec la même taille."""
    from libsigma import image_visu as iv
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    from osgeo import gdal

    layers = []
    for p in liste_chemins_rgb:
        ds = gdal.Open(p)
        bande = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        bande[bande <= 0] = np.nan 
        layers.append(bande)
    
    img_rgb = np.nan_to_num(np.dstack(layers))
    
    ds_map = gdal.Open(chemin_carte)
    map_data = ds_map.ReadAsArray()
    map_masked = np.ma.masked_where(map_data <= 0, map_data)
    
    # MODIFICATION : Suppression de width_ratios et ajustement de figsize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Image Vraies Couleurs
    img_8bit = iv.rescale_to_8bits(
        img_rgb, stretch='percentile', percentiles=[2, 98]
    )
    ax1.imshow(img_8bit)
    ax1.set_title("Image Sentinel-2", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Carte Classée
    colors = ['#A9A9A9', '#7CFC00', '#9370DB', '#006400']
    custom_cmap = ListedColormap(colors)
    im = ax2.imshow(
        map_masked, cmap=custom_cmap, vmin=1, vmax=4, interpolation='none'
    )
    ax2.set_title("Carte de Végétation (Random Forest)", fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # La colorbar est liée à ax2 mais ne change plus la taille du subplot
    cbar = fig.colorbar(im, ax=ax2, ticks=[1, 2, 3, 4], shrink=0.6)
    cbar.ax.set_yticklabels(['Sol Nu', 'Herbe', 'Landes', 'Arbre'])
    
    plt.tight_layout()
    plt.show()