import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import re

# -------------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------------
DATA_DIR = "Data"
TRAIN_RATIO = 0.7
LABELS_FILE = "labels.txt"
CHARS = "0123456789.,"



def clean_prediction(text):
    """Ne garde que les caractères autorisés (chiffres, point, virgule)."""
    return "".join(c for c in text if c in CHARS)

# -------------------------------------------------------------------------
# 2. Chargement des images et des labels (depuis le nom de fichier)
# -------------------------------------------------------------------------
all_files = glob.glob(os.path.join(DATA_DIR, "*.*"))
valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]

if not image_files:
    raise ValueError(f"Aucune image trouvée dans '{DATA_DIR}'.")

print(f"Nombre d'images trouvées : {len(image_files)}")

def load_labels(labels_path):
    """Charge les labels depuis un fichier texte 'nom_fichier,label'."""
    labels = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, label = line.split(",", 1)
            labels[filename] = label
    return labels

labels_dict = load_labels(LABELS_FILE)

def get_label(path):
    filename = os.path.basename(path)
    return labels_dict.get(filename, "")


# -------------------------------------------------------------------------
# 4. Split train / test
# -------------------------------------------------------------------------
random.seed(42)
shuffled = image_files.copy()
random.shuffle(shuffled)

split_idx = int(TRAIN_RATIO * len(shuffled))
train_files = shuffled[:split_idx]
test_files = shuffled[split_idx:]

print(f"Train : {len(train_files)} images | Test : {len(test_files)} images")


# -------------------------------------------------------------------------
# 3. Initialisation de PaddleOCR
# -------------------------------------------------------------------------
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="en",
    device="cpu",
    enable_mkldnn=False
)
# -------------------------------------------------------------------------
# 4. Inférence sur toutes les images
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# 6. Fonction d'évaluation (réutilisable pour train et test)
# -------------------------------------------------------------------------

def preprocess(img):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # agrandir proprement
    img = cv2.GaussianBlur(img, (3,3), 0)  # réduire le bruit de pixellisation
    return img


def evaluate(files, split_name):
    print(f"\n--- Reconnaissance sur le set {split_name} ---")
    results = []
    for path in files:
        true_label = get_label(path)
        img = cv2.imread(path)
        img = preprocess(img)

        ocr_result = ocr.predict(img)
        raw_pred = ocr_result[0]["rec_texts"][0] if ocr_result and ocr_result[0]["rec_texts"] else ""
        pred_label = clean_prediction(raw_pred)

        results.append((path, true_label, pred_label))
        match = "✓" if pred_label == true_label else "✗"
        print(f"{match} {os.path.basename(path):20s} | Vrai: '{true_label}' | Prédit: '{pred_label}'")

    correct = sum(1 for _, true, pred in results if true == pred)
    accuracy = (correct / len(results)) * 100 if results else 0
    print(f"Précision {split_name} : {accuracy:.1f}% ({correct}/{len(results)})")
    return results, accuracy

# -------------------------------------------------------------------------
# 7. Évaluation sur train et test séparément
# -------------------------------------------------------------------------
train_results, train_accuracy = evaluate(train_files, "TRAIN")
test_results, test_accuracy = evaluate(test_files, "TEST")

print(f"\n=== Résumé ===")
print(f"Précision train : {train_accuracy:.1f}%")
print(f"Précision test  : {test_accuracy:.1f}%")

# -------------------------------------------------------------------------
# 8. Affichage des résultats du set test
# -------------------------------------------------------------------------
import math

n_display = len(test_results)
n_cols = 5
n_rows = math.ceil(n_display / n_cols)

fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))

for i, (path, true_label, pred_label) in enumerate(test_results):
    img_disp = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ax = plt.subplot(n_rows, n_cols, i + 1)
    ax.imshow(img_disp)
    ax.set_title(
        f"'{true_label}'/'{pred_label}'",
        color="green" if true_label == pred_label else "red",
        fontsize=9
    )
    ax.axis("off")

plt.suptitle(f"Résultats sur le set TEST — Précision : {test_accuracy:.1f}%")
plt.tight_layout()
plt.show()