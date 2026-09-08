# ============================================================
# 01_prepare_dataset.py
# Préparation du dataset PaddleOCR
# ============================================================

import os
import shutil
import random

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

DATA_DIR = "Data"
LABELS_FILE = "labels.txt"

OUTPUT_DIR = "train_data"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

# ------------------------------------------------------------
# LECTURE DES LABELS
# ------------------------------------------------------------

samples = []

with open(LABELS_FILE, "r", encoding="utf-8") as f:

    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        # On utilise split(",", 1)
        # afin de ne couper qu'une seule fois.
        parts = line.split(",", 1)

        if len(parts) != 2:
            print(
                f"[WARNING] Ligne {line_number} ignorée : {line}"
            )
            continue

        filename = parts[0].strip()
        label = parts[1].strip()

        image_path = os.path.join(DATA_DIR, filename)

        if not os.path.exists(image_path):
            print(
                f"[WARNING] Image introuvable : {image_path}"
            )
            continue

        # Vérification : uniquement des chiffres
        if not label.isdigit():
            print(
                f"[WARNING] Label non numérique : "
                f"{filename} -> {label}"
            )
            continue

        samples.append((filename, label))

# ------------------------------------------------------------
# MELANGE
# ------------------------------------------------------------

random.shuffle(samples)

n = len(samples)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_samples = samples[:n_train]
val_samples = samples[n_train:n_train + n_val]
test_samples = samples[n_train + n_val:]

print("\n============================================")
print("DATASET")
print("============================================")

print(f"Nombre total : {len(samples)}")
print(f"Train        : {len(train_samples)}")
print(f"Validation   : {len(val_samples)}")
print(f"Test         : {len(test_samples)}")

# ------------------------------------------------------------
# COPIE DES IMAGES
# ------------------------------------------------------------

def copy_images(samples, split):

    destination_dir = os.path.join(
        OUTPUT_DIR,
        split
    )

    for filename, label in samples:

        source = os.path.join(
            DATA_DIR,
            filename
        )

        destination = os.path.join(
            destination_dir,
            filename
        )

        shutil.copy2(source, destination)


copy_images(train_samples, "train")
copy_images(val_samples, "val")
copy_images(test_samples, "test")

# ------------------------------------------------------------
# CREATION DES FICHIERS TXT PADDLEOCR
# ------------------------------------------------------------

def write_label_file(samples, split):

    filepath = os.path.join(
        OUTPUT_DIR,
        f"{split}_list.txt"
    )

    with open(filepath, "w", encoding="utf-8") as f:

        for filename, label in samples:

            # Format PaddleOCR :
            # chemin_relatif<TAB>label

            relative_path = os.path.join(
                split,
                filename
            )

            f.write(
                f"{relative_path}\t{label}\n"
            )

    print(f"Créé : {filepath}")


write_label_file(train_samples, "train")
write_label_file(val_samples, "val")
write_label_file(test_samples, "test")

# ------------------------------------------------------------
# RESUME
# ------------------------------------------------------------

print("\n============================================")
print("PREPARATION TERMINEE")
print("============================================")

print(f"\nDataset créé dans : {OUTPUT_DIR}/")

print("""
Structure :

train_data/
├── train/
├── val/
├── test/
├── train_list.txt
├── val_list.txt
└── test_list.txt
""")

print("Tu peux maintenant lancer :")
print("02_augment_dataset.py")