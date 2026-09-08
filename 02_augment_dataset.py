# ============================================================
# 02_augment_dataset.py
# Augmentation des images d'entraînement
# + création de train_augmented_list.txt
# ============================================================

import os
import cv2
import random
import numpy as np

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

TRAIN_DIR = "train_data/train"

TRAIN_LIST = "train_data/train_list.txt"

AUGMENTED_LIST = "train_data/train_augmented_list.txt"

NB_AUGMENTATIONS = 8

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ------------------------------------------------------------
# FONCTIONS D'AUGMENTATION
# ------------------------------------------------------------

def change_brightness(image, value):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    v = np.clip(
        v.astype(np.int16) + value,
        0,
        255
    ).astype(np.uint8)

    hsv = cv2.merge([h, s, v])

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


def change_contrast(image, factor):

    result = cv2.convertScaleAbs(
        image,
        alpha=factor,
        beta=0
    )

    return result


def add_gaussian_noise(image):

    noise = np.random.normal(
        0,
        5,
        image.shape
    )

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return noisy


def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (
        w // 2,
        h // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )


def blur_image(image):

    return cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )


def resize_scale(image, scale):

    h, w = image.shape[:2]

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )

    # Retour à la taille originale
    result = cv2.resize(
        resized,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    return result


# ------------------------------------------------------------
# AUGMENTATION ALEATOIRE
# ------------------------------------------------------------

def augment(image):

    result = image.copy()

    operation = random.choice([
        "brightness",
        "contrast",
        "noise",
        "rotation",
        "blur",
        "scale",
        "brightness_contrast",
        "rotation_blur"
    ])

    if operation == "brightness":

        value = random.randint(
            -25,
            25
        )

        result = change_brightness(
            result,
            value
        )

    elif operation == "contrast":

        factor = random.uniform(
            0.8,
            1.2
        )

        result = change_contrast(
            result,
            factor
        )

    elif operation == "noise":

        result = add_gaussian_noise(
            result
        )

    elif operation == "rotation":

        angle = random.uniform(
            -5,
            5
        )

        result = rotate_image(
            result,
            angle
        )

    elif operation == "blur":

        result = blur_image(
            result
        )

    elif operation == "scale":

        scale = random.uniform(
            0.90,
            1.10
        )

        result = resize_scale(
            result,
            scale
        )

    elif operation == "brightness_contrast":

        result = change_brightness(
            result,
            random.randint(-20, 20)
        )

        result = change_contrast(
            result,
            random.uniform(0.85, 1.15)
        )

    elif operation == "rotation_blur":

        result = rotate_image(
            result,
            random.uniform(-4, 4)
        )

        result = blur_image(
            result
        )

    return result


# ------------------------------------------------------------
# LECTURE DES LABELS
# ------------------------------------------------------------

print("\n============================================")
print("LECTURE DU DATASET")
print("============================================")

if not os.path.exists(TRAIN_LIST):

    print(
        f"[ERREUR] Fichier introuvable : "
        f"{TRAIN_LIST}"
    )

    print(
        "\nLance d'abord :"
    )

    print(
        "01_prepare_dataset.py"
    )

    raise SystemExit


labels = {}

with open(
    TRAIN_LIST,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        # Format attendu :
        # train/1.png    1

        parts = line.split("\t")

        if len(parts) != 2:

            print(
                f"[WARNING] Ligne ignorée : {line}"
            )

            continue

        image_path = parts[0]
        label = parts[1]

        filename = os.path.basename(
            image_path
        )

        labels[filename] = label


print(
    f"Labels trouvés : {len(labels)}"
)


# ------------------------------------------------------------
# RECHERCHE DES IMAGES
# ------------------------------------------------------------

extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp"
)

files = [
    f
    for f in os.listdir(TRAIN_DIR)
    if f.lower().endswith(extensions)
    and "_aug_" not in f
]

files.sort()

print(
    f"Images originales : {len(files)}"
)

print(
    f"Augmentations par image : "
    f"{NB_AUGMENTATIONS}"
)


# ------------------------------------------------------------
# AUGMENTATION
# ------------------------------------------------------------

total_generated = 0

augmented_filenames = []


for index, filename in enumerate(files, 1):

    path = os.path.join(
        TRAIN_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is None:

        print(
            f"[WARNING] Impossible de lire : "
            f"{filename}"
        )

        continue

    if filename not in labels:

        print(
            f"[WARNING] Label introuvable pour : "
            f"{filename}"
        )

        continue

    base_name = os.path.splitext(
        filename
    )[0]

    extension = os.path.splitext(
        filename
    )[1]

    for i in range(
        1,
        NB_AUGMENTATIONS + 1
    ):

        augmented = augment(
            image
        )

        output_name = (
            f"{base_name}_aug_{i}"
            f"{extension}"
        )

        output_path = os.path.join(
            TRAIN_DIR,
            output_name
        )

        success = cv2.imwrite(
            output_path,
            augmented
        )

        if success:

            augmented_filenames.append(
                output_name
            )

            total_generated += 1

    print(
        f"[{index}/{len(files)}] "
        f"{filename} -> "
        f"{NB_AUGMENTATIONS} augmentations"
    )


# ------------------------------------------------------------
# CREATION DE train_augmented_list.txt
# ------------------------------------------------------------

print("\n============================================")
print("CREATION DE LA LISTE D'ENTRAINEMENT")
print("============================================")


with open(
    AUGMENTED_LIST,
    "w",
    encoding="utf-8"
) as f:

    # --------------------------------------------------------
    # Images originales
    # --------------------------------------------------------

    for filename in files:

        if filename not in labels:
            continue

        label = labels[filename]

        f.write(
            f"train/{filename}\t{label}\n"
        )

    # --------------------------------------------------------
    # Images augmentées
    # --------------------------------------------------------

    for filename in augmented_filenames:

        # Retrouver le nom de l'image originale
        # Exemple :
        # 12_aug_3.png -> 12.png

        base_name = filename.split(
            "_aug_"
        )[0]

        extension = os.path.splitext(
            filename
        )[1]

        original_filename = (
            base_name + extension
        )

        if original_filename not in labels:

            print(
                f"[WARNING] Label introuvable pour "
                f"{filename}"
            )

            continue

        label = labels[
            original_filename
        ]

        f.write(
            f"train/{filename}\t{label}\n"
        )


# ------------------------------------------------------------
# VERIFICATION
# ------------------------------------------------------------

with open(
    AUGMENTED_LIST,
    "r",
    encoding="utf-8"
) as f:

    total_lines = sum(
        1
        for line in f
        if line.strip()
    )


# ------------------------------------------------------------
# RESULTATS
# ------------------------------------------------------------

print("\n============================================")
print("AUGMENTATION TERMINEE")
print("============================================")

print(
    f"Images originales : {len(files)}"
)

print(
    f"Images augmentées : {total_generated}"
)

print(
    f"Total images d'entraînement : "
    f"{total_lines}"
)

print(
    f"\nListe créée :"
)

print(
    AUGMENTED_LIST
)

print("\n============================================")
print("PRET POUR L'ENTRAINEMENT")
print("============================================")

print(
    "\nTu peux maintenant lancer :"
)

print(
    "03_train_ocr.py"
)