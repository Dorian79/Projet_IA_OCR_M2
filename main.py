import os
import glob
import random
import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import numpy as np
import pandas as pd
from collections import Counter
import math


# =========================================================================
# 1. CONFIGURATION
# =========================================================================

DATA_DIR = "Data2"
TRAIN_RATIO = 0.7
LABELS_FILE = "Data2/labels.txt"

# Caractères autorisés dans les prédictions
CHARS = "0123456789.,"

# Seed pour avoir toujours le même découpage train/test
RANDOM_SEED = 42


# =========================================================================
# 2. NETTOYAGE DES PRÉDICTIONS
# =========================================================================

def clean_prediction(text):
    """
    Ne conserve que les caractères autorisés :
    chiffres, point et virgule.
    """
    if text is None:
        return ""

    return "".join(
        c for c in str(text)
        if c in CHARS
    )


# =========================================================================
# 3. CHARGEMENT DES IMAGES
# =========================================================================

all_files = glob.glob(
    os.path.join(DATA_DIR, "*.*")
)

valid_exts = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".bmp"
}

image_files = [
    f for f in all_files
    if os.path.splitext(f)[1].lower() in valid_exts
]

if not image_files:
    raise ValueError(
        f"Aucune image trouvée dans '{DATA_DIR}'."
    )

print(
    f"Nombre d'images trouvées : "
    f"{len(image_files)}"
)


# =========================================================================
# 4. CHARGEMENT DES LABELS
# =========================================================================

def load_labels(labels_path):
    """
    Charge les labels depuis un fichier texte.

    Format attendu :

        image1.png,123
        image2.png,45.6
        image3.png,789,12

    Le split est effectué uniquement sur la première virgule.
    """

    labels = {}

    with open(
        labels_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            if "," not in line:
                print(
                    f"⚠ Ligne {line_number} ignorée : "
                    f"format incorrect -> {line}"
                )
                continue

            filename, label = line.split(",", 1)

            filename = filename.strip()
            label = label.strip()

            labels[filename] = label

    return labels


labels_dict = load_labels(
    LABELS_FILE
)


def get_label(path):
    """
    Retourne le label associé au nom du fichier.
    """

    filename = os.path.basename(path)

    return labels_dict.get(
        filename,
        ""
    )


# =========================================================================
# 5. VÉRIFICATION DES LABELS
# =========================================================================

missing_labels = [
    f
    for f in image_files
    if get_label(f) == ""
]

if missing_labels:

    print(
        f"\n⚠ Attention : "
        f"{len(missing_labels)} image(s) "
        f"n'ont pas de label."
    )

    for path in missing_labels[:10]:
        print(
            f"   - {os.path.basename(path)}"
        )

    if len(missing_labels) > 10:
        print(
            f"   ... et "
            f"{len(missing_labels) - 10} autre(s)"
        )


# =========================================================================
# 6. SPLIT TRAIN / TEST
# =========================================================================

random.seed(RANDOM_SEED)

shuffled = image_files.copy()

random.shuffle(shuffled)

split_idx = int(
    TRAIN_RATIO * len(shuffled)
)

train_files = shuffled[:split_idx]

test_files = shuffled[split_idx:]

print(
    f"\nTrain : {len(train_files)} images "
    f"| Test : {len(test_files)} images"
)


# =========================================================================
# 7. INITIALISATION DE PADDLEOCR
# =========================================================================

print("\nInitialisation de PaddleOCR...")

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    lang="en",
    device="cpu",
    enable_mkldnn=False
)

print("PaddleOCR prêt.")


# =========================================================================
# 8. PRÉTRAITEMENT
# =========================================================================

def preprocess(img):
    """
    Prétraitement des images avant OCR.
    """

    img = cv2.resize(
        img,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0
    )

    return img


# =========================================================================
# 9. DISTANCE DE LEVENSHTEIN
# =========================================================================

def levenshtein_distance(s1, s2):
    """
    Calcule la distance d'édition de Levenshtein
    entre deux chaînes.

    Elle correspond au nombre minimal de :
        - suppressions
        - insertions
        - substitutions

    nécessaires pour transformer s1 en s2.
    """

    if s1 == s2:
        return 0

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous = list(
        range(len(s2) + 1)
    )

    for i, c1 in enumerate(
        s1,
        start=1
    ):

        current = [i]

        for j, c2 in enumerate(
            s2,
            start=1
        ):

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (c1 != c2)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


# =========================================================================
# 10. CER - CHARACTER ERROR RATE
# =========================================================================

def cer(true_text, predicted_text):
    """
    Character Error Rate.

    CER = distance d'édition / longueur du texte réel.

    Plus le CER est proche de 0, meilleur est le résultat.
    """

    if len(true_text) == 0:

        if len(predicted_text) == 0:
            return 0.0

        return 1.0

    distance = levenshtein_distance(
        true_text,
        predicted_text
    )

    return distance / len(true_text)


# =========================================================================
# 11. ACCURACY CARACTÈRE
# =========================================================================

def character_accuracy(
    true_text,
    predicted_text
):
    """
    Approximation de l'accuracy caractère
    à partir de la distance d'édition.
    """

    if len(true_text) == 0:

        if len(predicted_text) == 0:
            return 1.0

        return 0.0

    distance = levenshtein_distance(
        true_text,
        predicted_text
    )

    return max(
        0.0,
        1.0 - distance / len(true_text)
    )


# =========================================================================
# 12. CATÉGORIE DE LONGUEUR
# =========================================================================

def get_length_category(length):

    if length == 0:
        return "0"

    if length == 1:
        return "1"

    if length == 2:
        return "2"

    if length == 3:
        return "3"

    if length == 4:
        return "4"

    return "5+"


# =========================================================================
# 13. INTERVALLE DE CONFIANCE BOOTSTRAP
# =========================================================================

def bootstrap_confidence_interval(
    values,
    statistic=np.mean,
    n_bootstrap=2000,
    confidence=0.95,
    random_state=42
):
    """
    Calcule un intervalle de confiance bootstrap.
    """

    values = np.asarray(values)

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_values = []

    for _ in range(n_bootstrap):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_values.append(
            statistic(sample)
        )

    alpha = 1 - confidence

    lower = np.percentile(
        bootstrap_values,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_values,
        100 * (1 - alpha / 2)
    )

    return lower, upper


# =========================================================================
# 14. ÉVALUATION OCR
# =========================================================================

def evaluate(files, split_name):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"RECONNAISSANCE SUR LE SET {split_name}"
    )

    print(
        f"{'=' * 70}"
    )

    results = []

    for index, path in enumerate(
        files,
        start=1
    ):

        true_label = get_label(path)

        img = cv2.imread(path)

        if img is None:

            print(
                f"⚠ Impossible de lire : "
                f"{path}"
            )

            continue

        # -------------------------------------------------------------
        # Prétraitement
        # -------------------------------------------------------------

        img_processed = preprocess(img)

        # -------------------------------------------------------------
        # OCR
        # -------------------------------------------------------------

        try:

            ocr_result = ocr.predict(
                img_processed
            )

        except Exception as error:

            print(
                f"⚠ Erreur OCR pour "
                f"{os.path.basename(path)} : "
                f"{error}"
            )

            ocr_result = None

        # -------------------------------------------------------------
        # Récupération de la prédiction
        # -------------------------------------------------------------

        raw_pred = ""

        confidence = np.nan

        if ocr_result:

            result = ocr_result[0]

            # Textes reconnus
            rec_texts = result.get(
                "rec_texts",
                []
            )

            # Scores de confiance
            rec_scores = result.get(
                "rec_scores",
                []
            )

            if rec_texts:

                raw_pred = rec_texts[0]

            if rec_scores:

                try:

                    confidence = float(
                        rec_scores[0]
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    confidence = np.nan

        # -------------------------------------------------------------
        # Nettoyage
        # -------------------------------------------------------------

        pred_label = clean_prediction(
            raw_pred
        )

        # -------------------------------------------------------------
        # Métriques
        # -------------------------------------------------------------

        exact_match = int(
            pred_label == true_label
        )

        edit_distance = (
            levenshtein_distance(
                true_label,
                pred_label
            )
        )

        current_cer = cer(
            true_label,
            pred_label
        )

        char_acc = character_accuracy(
            true_label,
            pred_label
        )

        true_length = len(
            true_label
        )

        pred_length = len(
            pred_label
        )

        length_error = (
            pred_length - true_length
        )

        length_category = (
            get_length_category(
                true_length
            )
        )

        # -------------------------------------------------------------
        # Stockage
        # -------------------------------------------------------------

        results.append({

            "file": os.path.basename(path),

            "true": true_label,

            "pred": pred_label,

            "correct": exact_match,

            "confidence": confidence,

            "edit_distance": edit_distance,

            "cer": current_cer,

            "character_accuracy": char_acc,

            "true_length": true_length,

            "pred_length": pred_length,

            "length_error": length_error,

            "length_category": length_category

        })

        # -------------------------------------------------------------
        # Affichage
        # -------------------------------------------------------------

        match = (
            "✓"
            if exact_match
            else "✗"
        )

        confidence_text = (
            f"{confidence:.3f}"
            if not np.isnan(confidence)
            else "N/A"
        )

        print(
            f"{match} "
            f"[{index:4d}/{len(files):4d}] "
            f"{os.path.basename(path):25s} | "
            f"Vrai: '{true_label}' | "
            f"Prédit: '{pred_label}' | "
            f"Conf: {confidence_text} | "
            f"CER: {current_cer:.3f}"
        )

    # =================================================================
    # DATAFRAME
    # =================================================================

    df = pd.DataFrame(
        results
    )

    if df.empty:

        print(
            "Aucun résultat disponible."
        )

        return df, {}

    # =================================================================
    # MÉTRIQUES GLOBALES
    # =================================================================

    accuracy = df[
        "correct"
    ].mean()

    mean_cer = df[
        "cer"
    ].mean()

    mean_edit_distance = df[
        "edit_distance"
    ].mean()

    mean_character_accuracy = df[
        "character_accuracy"
    ].mean()

    mean_confidence = df[
        "confidence"
    ].mean()

    # =================================================================
    # INTERVALLE DE CONFIANCE
    # =================================================================

    accuracy_ci_low, accuracy_ci_high = (
        bootstrap_confidence_interval(
            df["correct"].values,
            statistic=np.mean
        )
    )

    # =================================================================
    # LONGUEUR
    # =================================================================

    exact_length_accuracy = (
        df["length_error"] == 0
    ).mean()

    too_short = (
        df["length_error"] < 0
    ).mean()

    too_long = (
        df["length_error"] > 0
    ).mean()

    # =================================================================
    # STATISTIQUES
    # =================================================================

    metrics = {

        "accuracy": accuracy,

        "accuracy_ci_low":
            accuracy_ci_low,

        "accuracy_ci_high":
            accuracy_ci_high,

        "mean_cer":
            mean_cer,

        "mean_edit_distance":
            mean_edit_distance,

        "character_accuracy":
            mean_character_accuracy,

        "mean_confidence":
            mean_confidence,

        "length_accuracy":
            exact_length_accuracy,

        "too_short_rate":
            too_short,

        "too_long_rate":
            too_long,

        "n_samples":
            len(df)
    }

    # =================================================================
    # AFFICHAGE
    # =================================================================

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"STATISTIQUES — {split_name}"
    )

    print(
        f"{'=' * 70}"
    )

    print(
        f"Nombre d'images       : "
        f"{len(df)}"
    )

    print(
        f"Accuracy exacte       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"IC 95% accuracy       : "
        f"[{accuracy_ci_low * 100:.2f}%, "
        f"{accuracy_ci_high * 100:.2f}%]"
    )

    print(
        f"CER moyen             : "
        f"{mean_cer * 100:.2f}%"
    )

    print(
        f"Accuracy caractère    : "
        f"{mean_character_accuracy * 100:.2f}%"
    )

    print(
        f"Distance édition moy. : "
        f"{mean_edit_distance:.3f}"
    )

    print(
        f"Confiance OCR moyenne : "
        f"{mean_confidence:.3f}"
    )

    print(
        f"Longueur correcte     : "
        f"{exact_length_accuracy * 100:.2f}%"
    )

    print(
        f"Prédiction trop courte: "
        f"{too_short * 100:.2f}%"
    )

    print(
        f"Prédiction trop longue: "
        f"{too_long * 100:.2f}%"
    )

    return df, metrics


# =========================================================================
# 15. ANALYSE PAR LONGUEUR
# =========================================================================

def analyze_by_length(df):

    print(
        f"\n{'=' * 70}"
    )

    print(
        "STATISTIQUES PAR LONGUEUR"
    )

    print(
        f"{'=' * 70}"
    )

    if df.empty:
        return pd.DataFrame()

    table = (
        df
        .groupby("length_category")
        .agg(
            samples=("correct", "count"),

            accuracy=("correct", "mean"),

            cer=("cer", "mean"),

            character_accuracy=(
                "character_accuracy",
                "mean"
            ),

            confidence=(
                "confidence",
                "mean"
            )
        )
        .reset_index()
    )

    # Conversion en pourcentage
    table["accuracy"] *= 100
    table["cer"] *= 100
    table["character_accuracy"] *= 100

    table = table.round(2)

    print(
        table.to_string(
            index=False
        )
    )

    return table


# =========================================================================
# 16. ANALYSE PAR CHIFFRE
# =========================================================================

def analyze_digits(df):

    print(
        f"\n{'=' * 70}"
    )

    print(
        "STATISTIQUES PAR CHIFFRE"
    )

    print(
        f"{'=' * 70}"
    )

    stats = []

    for digit in "0123456789":

        relevant = df[
            df["true"].str.contains(
                digit,
                regex=False
            )
        ]

        if len(relevant) == 0:
            continue

        stats.append({

            "digit": digit,

            "samples": len(relevant),

            "exact_accuracy":
                relevant["correct"].mean(),

            "mean_cer":
                relevant["cer"].mean(),

            "mean_confidence":
                relevant["confidence"].mean()
        })

    table = pd.DataFrame(
        stats
    )

    if not table.empty:

        table["exact_accuracy"] *= 100

        table["mean_cer"] *= 100

        table = table.round(2)

        print(
            table.to_string(
                index=False
            )
        )

    return table


# =========================================================================
# 17. ANALYSE DES ERREURS DE CARACTÈRES
# =========================================================================

def get_character_errors(
    true_text,
    pred_text
):
    """
    Compare deux chaînes et récupère
    les substitutions caractère par caractère.

    Exemple :

        vrai    = 123
        prédit  = 128

        -> 3 devient 8
    """

    errors = []

    i = 0
    j = 0

    while (
        i < len(true_text)
        and
        j < len(pred_text)
    ):

        if true_text[i] == pred_text[j]:

            i += 1
            j += 1

        else:

            errors.append(
                (
                    true_text[i],
                    pred_text[j]
                )
            )

            i += 1
            j += 1

    return errors


def analyze_character_errors(df):

    print(
        f"\n{'=' * 70}"
    )

    print(
        "ERREURS DE CARACTÈRES"
    )

    print(
        f"{'=' * 70}"
    )

    counter = Counter()

    for _, row in df.iterrows():

        true_text = row["true"]

        pred_text = row["pred"]

        errors = get_character_errors(
            true_text,
            pred_text
        )

        for (
            true_char,
            pred_char
        ) in errors:

            counter[
                (
                    true_char,
                    pred_char
                )
            ] += 1

    rows = []

    for (
        (
            true_char,
            pred_char
        ),
        count
    ) in counter.most_common():

        rows.append({

            "true": true_char,

            "predicted": pred_char,

            "count": count

        })

    table = pd.DataFrame(
        rows
    )

    if not table.empty:

        print(
            table
            .head(20)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "Aucune erreur de "
            "substitution détectée."
        )

    return table


# =========================================================================
# 18. ANALYSE DES ERREURS À FAIBLE CONFIANCE
# =========================================================================

def analyze_low_confidence_errors(
    df,
    threshold=0.80
):

    print(
        f"\n{'=' * 70}"
    )

    print(
        f"ERREURS À FAIBLE CONFIANCE "
        f"(< {threshold})"
    )

    print(
        f"{'=' * 70}"
    )

    valid_confidence = df[
        df["confidence"].notna()
    ]

    low_conf = valid_confidence[
        valid_confidence["confidence"]
        < threshold
    ]

    errors = low_conf[
        low_conf["correct"] == 0
    ]

    print(
        f"Images avec confiance < "
        f"{threshold} : "
        f"{len(low_conf)}"
    )

    print(
        f"Erreurs parmi celles-ci : "
        f"{len(errors)}"
    )

    if len(low_conf) > 0:

        error_rate = (
            len(errors)
            /
            len(low_conf)
        )

        print(
            f"Taux d'erreur : "
            f"{error_rate * 100:.2f}%"
        )

    return errors


# =========================================================================
# 19. GRAPHIQUES
# =========================================================================

def plot_ocr_statistics(
    df,
    split_name="TEST"
):

    if df.empty:
        print(
            "Impossible de générer "
            "les graphiques."
        )
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    # -----------------------------------------------------------------
    # 1. Distribution du CER
    # -----------------------------------------------------------------

    axes[0, 0].hist(
        df["cer"],
        bins=20
    )

    axes[0, 0].set_title(
        f"Distribution du CER — {split_name}"
    )

    axes[0, 0].set_xlabel(
        "CER"
    )

    axes[0, 0].set_ylabel(
        "Nombre d'images"
    )

    # -----------------------------------------------------------------
    # 2. Distribution de la confiance
    # -----------------------------------------------------------------

    confidence_values = (
        df["confidence"]
        .dropna()
    )

    if len(confidence_values) > 0:

        axes[0, 1].hist(
            confidence_values,
            bins=20
        )

    axes[0, 1].set_title(
        "Distribution de la confiance OCR"
    )

    axes[0, 1].set_xlabel(
        "Confiance"
    )

    axes[0, 1].set_ylabel(
        "Nombre d'images"
    )

    # -----------------------------------------------------------------
    # 3. Accuracy selon la longueur
    # -----------------------------------------------------------------

    grouped = (
        df
        .groupby("length_category")[
            "correct"
        ]
        .mean()
    )

    axes[1, 0].bar(
        grouped.index,
        grouped.values * 100
    )

    axes[1, 0].set_title(
        "Accuracy selon la longueur"
    )

    axes[1, 0].set_xlabel(
        "Nombre de caractères"
    )

    axes[1, 0].set_ylabel(
        "Accuracy (%)"
    )

    axes[1, 0].set_ylim(
        0,
        100
    )

    # -----------------------------------------------------------------
    # 4. Confiance vs CER
    # -----------------------------------------------------------------

    if len(confidence_values) > 0:

        valid_rows = df[
            df["confidence"].notna()
        ]

        axes[1, 1].scatter(
            valid_rows["confidence"],
            valid_rows["cer"],
            alpha=0.5
        )

    axes[1, 1].set_title(
        "Confiance OCR vs CER"
    )

    axes[1, 1].set_xlabel(
        "Confiance"
    )

    axes[1, 1].set_ylabel(
        "CER"
    )

    plt.suptitle(
        f"Analyse des performances OCR — "
        f"{split_name}"
    )

    plt.tight_layout()

    plt.show()


# =========================================================================
# 20. AFFICHAGE DES IMAGES DE TEST
# =========================================================================

def display_test_results(
    test_results,
    test_metrics
):

    if test_results.empty:
        return

    n_display = len(
        test_results
    )

    n_cols = 5

    n_rows = math.ceil(
        n_display / n_cols
    )

    fig = plt.figure(
        figsize=(
            n_cols * 2.5,
            n_rows * 2.5
        )
    )

    for i, row in test_results.iterrows():

        path = os.path.join(
            DATA_DIR,
            row["file"]
        )

        img = cv2.imread(
            path
        )

        if img is None:
            continue

        img_disp = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        ax = plt.subplot(
            n_rows,
            n_cols,
            i + 1
        )

        ax.imshow(
            img_disp
        )

        is_correct = (
            row["true"]
            ==
            row["pred"]
        )

        ax.set_title(
            f"V: '{row['true']}'\n"
            f"P: '{row['pred']}'",
            color=(
                "green"
                if is_correct
                else "red"
            ),
            fontsize=9
        )

        ax.axis(
            "off"
        )

    plt.suptitle(
        "Résultats sur le set TEST — "
        f"Accuracy : "
        f"{test_metrics['accuracy'] * 100:.1f}%"
    )

    plt.tight_layout()

    plt.show()


# =========================================================================
# 21. ÉVALUATION TRAIN
# =========================================================================

train_results, train_metrics = evaluate(
    train_files,
    "TRAIN"
)


# =========================================================================
# 22. ÉVALUATION TEST
# =========================================================================

test_results, test_metrics = evaluate(
    test_files,
    "TEST"
)


# =========================================================================
# 23. RÉSUMÉ GLOBAL
# =========================================================================

print(
    f"\n{'=' * 70}"
)

print(
    "RÉSUMÉ FINAL"
)

print(
    f"{'=' * 70}"
)

print(
    f"Train accuracy : "
    f"{train_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Test accuracy  : "
    f"{test_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Test CER       : "
    f"{test_metrics['mean_cer'] * 100:.2f}%"
)

print(
    f"Test char acc. : "
    f"{test_metrics['character_accuracy'] * 100:.2f}%"
)

print(
    f"Test distance  : "
    f"{test_metrics['mean_edit_distance']:.3f}"
)

print(
    f"Confiance OCR  : "
    f"{test_metrics['mean_confidence']:.3f}"
)


# =========================================================================
# 24. ANALYSE PAR LONGUEUR
# =========================================================================

train_length_stats = analyze_by_length(
    train_results
)

test_length_stats = analyze_by_length(
    test_results
)


# =========================================================================
# 25. ANALYSE PAR CHIFFRE
# =========================================================================

train_digit_stats = analyze_digits(
    train_results
)

test_digit_stats = analyze_digits(
    test_results
)


# =========================================================================
# 26. ANALYSE DES ERREURS DE CARACTÈRES
# =========================================================================

test_character_errors = (
    analyze_character_errors(
        test_results
    )
)


# =========================================================================
# 27. ANALYSE DES ERREURS À FAIBLE CONFIANCE
# =========================================================================

low_confidence_errors = (
    analyze_low_confidence_errors(
        test_results,
        threshold=0.80
    )
)


# =========================================================================
# 28. GRAPHIQUES
# =========================================================================

plot_ocr_statistics(
    test_results,
    "TEST"
)


# =========================================================================
# 29. AFFICHAGE DES IMAGES TEST
# =========================================================================

display_test_results(
    test_results,
    test_metrics
)


# =========================================================================
# 30. EXPORT DES RÉSULTATS
# =========================================================================

print(
    f"\n{'=' * 70}"
)

print(
    "EXPORT DES RÉSULTATS"
)

print(
    f"{'=' * 70}"
)


# Prédictions individuelles
test_results.to_csv(
    "ocr_test_predictions.csv",
    index=False,
    encoding="utf-8"
)

train_results.to_csv(
    "ocr_train_predictions.csv",
    index=False,
    encoding="utf-8"
)


# Statistiques par longueur
test_length_stats.to_csv(
    "ocr_test_statistics_by_length.csv",
    index=False,
    encoding="utf-8"
)


# Statistiques par chiffre
test_digit_stats.to_csv(
    "ocr_test_statistics_by_digit.csv",
    index=False,
    encoding="utf-8"
)


# Erreurs de caractères
test_character_errors.to_csv(
    "ocr_test_character_errors.csv",
    index=False,
    encoding="utf-8"
)


print(
    "Fichiers CSV générés :"
)

print(
    "  - ocr_test_predictions.csv"
)

print(
    "  - ocr_train_predictions.csv"
)

print(
    "  - ocr_test_statistics_by_length.csv"
)

print(
    "  - ocr_test_statistics_by_digit.csv"
)

print(
    "  - ocr_test_character_errors.csv"
)


# =========================================================================
# 31. FIN
# =========================================================================

print(
    f"\n{'=' * 70}"
)

print(
    "ÉVALUATION TERMINÉE"
)

print(
    f"{'=' * 70}"
)