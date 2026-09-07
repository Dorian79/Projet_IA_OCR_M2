print("Hello World")

import sys
import subprocess

def installer_paddleocr():
    """Vérifie et installe automatiquement les dépendances requises."""
    try:
        import paddleocr
        import cv2
        import numpy as np
        print("PaddleOCR et les dépendances sont déjà installés.")
    except ImportError:
        print("Installation des dépendances en cours...")
        commandes = [
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            [sys.executable, "-m", "pip", "install", "numpy<2.0.0"],
            [sys.executable, "-m", "pip", "install", "paddlepaddle", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cpu/"],
            [sys.executable, "-m", "pip", "install", "paddleocr", "opencv-python", "pillow"]
        ]
        for cmd in commandes:
            subprocess.check_call(cmd)
        print("Installation terminée !")

# Lancement de l'installation automatique
installer_paddleocr()

# -------------------------------------------------------------
# Votre code PaddleOCR (Reconnaissance seule)
# -------------------------------------------------------------
import cv2
from paddleocr import PaddleOCR

# Initialisation du modèle
ocr = PaddleOCR(use_textline_orientation=True, lang="en", device="cpu")

# Exemple d'appel sur un crop de chiffre :
# img = cv2.imread("votre_crop.png")
# resultat = ocr.ocr(img, det=False, cls=True)
# print(resultat)