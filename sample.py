import cv2
import os

def decouper_carte_yolo(chemin_carte, chemin_txt, dossier_sortie="Data2", marge_pixels=2):
    """
    Découpe une carte à partir d'un fichier .txt au format YOLO :
    label x_centre y_centre largeur hauteur (valeurs normalisées entre 0 et 1)
    """
    # 1. Charger la carte
    carte = cv2.imread(chemin_carte)
    if carte is None:
        print(f"Erreur : Impossible de charger l'image '{chemin_carte}'")
        return

    hauteur_carte, largeur_carte = carte.shape[:2]
    
    # 2. Préparer le dossier de sortie
    os.makedirs(dossier_sortie, exist_ok=True)
    chemin_labels_sortie = os.path.join(dossier_sortie, "labels.txt")
    fichier_labels = open(chemin_labels_sortie, "w", encoding="utf-8")

    # 3. Lire le fichier d'annotations
    if not os.path.exists(chemin_txt):
        print(f"Erreur : Fichier '{chemin_txt}' introuvable.")
        return

    with open(chemin_txt, "r", encoding="utf-8") as f:
        lignes = f.readlines()

    count = 0
    print(f"Découpage en cours (Taille de la carte : {largeur_carte}x{hauteur_carte})...")

    for ligne in lignes:
        ligne = ligne.strip()
        if not ligne:
            continue
            
        parts = ligne.split()
        if len(parts) >= 5:
            # Récupération des données du txt
            label = parts[0]
            x_centre_norm = float(parts[1])
            y_centre_norm = float(parts[2])
            w_norm = float(parts[3])
            h_norm = float(parts[4])

            # Conversion en pixels réels
            x_centre = x_centre_norm * largeur_carte
            y_centre = y_centre_norm * hauteur_carte
            w = w_norm * largeur_carte
            h = h_norm * hauteur_carte

            # Calcul du point haut-gauche et bas-droit (avec marge optionnelle)
            x_debut = int(x_centre - (w / 2)) - marge_pixels
            y_debut = int(y_centre - (h / 2)) - marge_pixels
            x_fin = int(x_centre + (w / 2)) + marge_pixels
            y_fin = int(y_centre + (h / 2)) + marge_pixels

            # Sécurité pour ne pas déborder de l'image
            x_debut = max(0, x_debut)
            y_debut = max(0, y_debut)
            x_fin = min(largeur_carte, x_fin)
            y_fin = min(hauteur_carte, y_fin)

            # Découper l'imagette
            crop = carte[y_debut:y_fin, x_debut:x_fin]

            # Ignorer si la boîte est vide
            if crop.size == 0 or x_fin <= x_debut or y_fin <= y_debut:
                continue

            # Sauvegarder
            nom_fichier = f"crop_{count:04d}.png"
            chemin_crop = os.path.join(dossier_sortie, nom_fichier)
            cv2.imwrite(chemin_crop, crop)

            # Écrire dans le nouveau labels.txt au format attendu par PaddleOCR
            fichier_labels.write(f"{nom_fichier},{label}\n")
            count += 1

    fichier_labels.close()
    print(f"Terminé ! {count} imagettes sauvegardées dans '{dossier_sortie}'.")

# ==========================================
# UTILISATION
# ==========================================
if __name__ == "__main__":
    # Renseignez les chemins vers vos fichiers
    FICHIER_CARTE = "carte2.jpg_map.jpg"  # L'image complète de la carte
    FICHIER_TXT = "carte2.jpg_map.txt"     # Le fichier texte avec les données YOLO
    
    decouper_carte_yolo(FICHIER_CARTE, FICHIER_TXT, dossier_sortie="Data2")