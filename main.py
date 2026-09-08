import os
import glob
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# -------------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------------
DATA_DIR = "Data"
CHARS = "0123456789.,"  # caractères autorisés (utilisé pour nettoyer les labels)

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

labels_dict = load_labels("labels.txt")

def get_label(path):
    filename = os.path.basename(path)
    return labels_dict.get(filename, "")

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
results = []  # liste de tuples (path, vrai_label, prediction)

print("\n--- Reconnaissance en cours ---")
for path in image_files:
    true_label = get_label(path)
    img = cv2.imread(path)

    ocr_result = ocr.predict(img)
    # Nouvelle API : le résultat est une liste de dicts avec la clé "rec_texts"
    pred_label = ocr_result[0]["rec_texts"][0] if ocr_result and ocr_result[0]["rec_texts"] else ""

    results.append((path, true_label, pred_label))
    match = "✓" if pred_label == true_label else "✗"
    print(f"{match} {os.path.basename(path):20s} | Vrai: '{true_label}' | Prédit: '{pred_label}'")

# -------------------------------------------------------------------------
# 5. Calcul de la précision globale
# -------------------------------------------------------------------------
correct = sum(1 for _, true, pred in results if true == pred)
accuracy = (correct / len(results)) * 100
print(f"\nPrécision globale : {accuracy:.1f}% ({correct}/{len(results)})")

# -------------------------------------------------------------------------
# 6. Affichage de quelques exemples
# -------------------------------------------------------------------------
n_display = min(8, len(results))
fig = plt.figure(figsize=(14, 6))

for i, (path, true_label, pred_label) in enumerate(results[:n_display]):
    img_disp = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ax = plt.subplot(2, 4, i + 1)
    ax.imshow(img_disp)
    ax.set_title(
        f"Vrai: '{true_label}'\nPrédit: '{pred_label}'",
        color="green" if true_label == pred_label else "red"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()