import sys
sys.path.append('../..')
import os
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, \
    accuracy_score
import geopandas as gpd

# personal librairies
import classification as cla
from libsigma import read_and_write as rw
import plots

def création_dossier():
    # Récupérer le dossier où se trouve le notebook
    notebook_dir = os.getcwd()

    # Définition des chemins
    img_path = os.path.join(notebook_dir, "img")
    results_path = os.path.join(notebook_dir, "results")
    figure_path = os.path.join(results_path, "figure")

    # Création des dossiers
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    print("Dossiers créés.")



def info_raster(FB3, FB5):
    """
    Ouvre deux images raster, affiche leurs dimensions et tailles de pixels.
    
    Parameters:
    FB3 (str): chemin vers la bande 3
    FB5 (str): chemin vers la bande 5
    """
    
    # Ouverture des images
    print(f"Ouverture de l'image B3 : {FB3}")
    b3 = rw.open_image(FB3, verbose=True)
    
    print(f"Ouverture de l'image B5 : {FB5}")
    b5 = rw.open_image(FB5, verbose=True)
    
    # Dimensions
    dim3 = nb_lignes3, nb_col3, nb_band3 = rw.get_image_dimension(b3, verbose=True)
    dim5 = nb_lignes5, nb_col5, nb_band5 = rw.get_image_dimension(b5, verbose=True)
    
    print(f"\nDimensions de B3 : lignes={nb_lignes3}, colonnes={nb_col3}, bandes={nb_band3}")
    print(f"Dimensions de B5 : lignes={nb_lignes5}, colonnes={nb_col5}, bandes={nb_band5}")
    
    # Taille des pixels (on prend B3 comme référence)
    psize_x, psize_y = rw.get_pixel_size(b3, verbose=True)
    print(f"Taille des pixels (B3) : psize_x={psize_x}, psize_y={psize_y}")
    
    # Retourner éventuellement les objets et infos
    return {
        "B3": {"raster": b3, "dimensions": dim3},
        "B5": {"raster": b5, "dimensions": dim5},
        "pixel_size": (psize_x, psize_y)
    }

