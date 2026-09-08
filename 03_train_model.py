# ============================================================
# 03_train_ocr.py
# Fine-tuning de PP-OCRv5_server_rec sur un alphabet numérique
# ============================================================

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import shutil
import yaml


# ============================================================
# CHEMINS DU PROJET
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

PYTHON = sys.executable

PADDLEOCR_DIR = PROJECT_DIR / "PaddleOCR"

PRETRAINED_DIR = PROJECT_DIR / "pretrained"
PRETRAINED_FILE = PRETRAINED_DIR / "PP-OCRv5_server_rec_pretrained.pdparams"

TRAIN_DATA_DIR = PROJECT_DIR / "train_data"

TRAIN_LIST = TRAIN_DATA_DIR / "train_augmented_list.txt"
VAL_LIST = TRAIN_DATA_DIR / "val_list.txt"
TEST_LIST = TRAIN_DATA_DIR / "test_list.txt"

OUTPUT_DIR = PROJECT_DIR / "output" / "digits_server_rec"

CONFIG_FILE = PROJECT_DIR / "PP-OCRv5_server_digits.yml"

DIGITS_DICT = (
    PADDLEOCR_DIR
    / "ppocr"
    / "utils"
    / "dict"
    / "digits_dict.txt"
)


# ============================================================
# PARAMETRES DU FINE-TUNING
# ============================================================

MODEL_NAME = "PP-OCRv5_server_rec"

ALPHABET = "0123456789"

EPOCHS = 30
LEARNING_RATE = 0.0001
BATCH_SIZE = 8

MAX_TEXT_LENGTH = 5

DEVICE = "cpu"


# ============================================================
# URL DU MODELE PRE-ENTRAINE
# ============================================================

PRETRAINED_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/"
    "paddlex/official_pretrained_model/"
    "PP-OCRv5_server_rec_pretrained.pdparams"
)


# ============================================================
# AFFICHAGE
# ============================================================

def titre(texte):
    print()
    print("=" * 60)
    print(texte)
    print("=" * 60)


# ============================================================
# VERIFICATION DE L'ENVIRONNEMENT
# ============================================================

def verifier_environnement():

    titre("ENVIRONNEMENT")

    try:
        import paddle

        print("PaddlePaddle :", paddle.__version__)

    except ImportError:
        print("[ERREUR] PaddlePaddle n'est pas installé.")
        sys.exit(1)

    try:
        import paddleocr

        print("PaddleOCR    :", paddleocr.__version__)

    except ImportError:
        print("[ERREUR] PaddleOCR n'est pas installé.")
        sys.exit(1)

    print("Python       :", sys.version.split()[0])
    print("Executable   :", PYTHON)


# ============================================================
# VERIFICATION / INSTALLATION DU DEPOT PADDLEOCR
# ============================================================

def verifier_paddleocr():

    titre("PADDLEOCR")

    if PADDLEOCR_DIR.exists():

        config_officielle = (
            PADDLEOCR_DIR
            / "configs"
            / "rec"
            / "PP-OCRv5"
            / "PP-OCRv5_server_rec.yml"
        )

        if config_officielle.exists():

            print("Dépôt PaddleOCR déjà présent.")
            print("Configuration officielle trouvée :")
            print(config_officielle)

            return config_officielle

        else:
            print(
                "[ERREUR] Le dossier PaddleOCR existe mais "
                "la configuration officielle est absente."
            )
            print(config_officielle)
            sys.exit(1)

    print("Dépôt PaddleOCR absent.")
    print("Clonage de la branche release/3.7...")

    commande = [
        "git",
        "clone",
        "--branch",
        "release/3.7",
        "--depth",
        "1",
        "https://github.com/PaddlePaddle/PaddleOCR.git",
        str(PADDLEOCR_DIR),
    ]

    try:
        subprocess.check_call(commande)

    except Exception as e:
        print()
        print("[ERREUR] Impossible de cloner PaddleOCR.")
        print(e)
        print()
        print("Vérifie que Git est installé et accessible depuis PowerShell.")
        sys.exit(1)

    config_officielle = (
        PADDLEOCR_DIR
        / "configs"
        / "rec"
        / "PP-OCRv5"
        / "PP-OCRv5_server_rec.yml"
    )

    if not config_officielle.exists():

        print("[ERREUR] Configuration officielle introuvable :")
        print(config_officielle)

        sys.exit(1)

    print("Configuration officielle trouvée :")
    print(config_officielle)

    return config_officielle


# ============================================================
# MODELE PRE-ENTRAINE
# ============================================================

def verifier_modele_preentraine():

    titre("MODELE PRE-ENTRAINE")

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if PRETRAINED_FILE.exists():

        print("Modèle pré-entraîné déjà présent.")
        print("Modèle :", PRETRAINED_FILE)

        return

    print("Modèle pré-entraîné absent.")
    print()
    print("Téléchargement du modèle PP-OCRv5_server_rec...")
    print(PRETRAINED_URL)
    print()

    try:

        urllib.request.urlretrieve(
            PRETRAINED_URL,
            PRETRAINED_FILE
        )

    except Exception as e:

        print()
        print("[ERREUR] Téléchargement du modèle impossible.")
        print(e)

        if PRETRAINED_FILE.exists():
            PRETRAINED_FILE.unlink()

        sys.exit(1)

    print()
    print("Téléchargement terminé.")
    print("Modèle :", PRETRAINED_FILE)


# ============================================================
# DICTIONNAIRE DE CHIFFRES
# ============================================================

def creer_dictionnaire():

    titre("DICTIONNAIRE")

    DIGITS_DICT.parent.mkdir(parents=True, exist_ok=True)

    with open(
        DIGITS_DICT,
        "w",
        encoding="utf-8"
    ) as f:

        for caractere in ALPHABET:
            f.write(caractere + "\n")

    print("Dictionnaire créé :")
    print(DIGITS_DICT)

    print()
    print("Alphabet :", ALPHABET)
    print("Nombre de caractères :", len(ALPHABET))


# ============================================================
# VERIFICATION DU DATASET
# ============================================================

def verifier_dataset():

    titre("DATASET")

    fichiers = [
        ("Train", TRAIN_LIST),
        ("Validation", VAL_LIST),
        ("Test", TEST_LIST),
    ]

    for nom, fichier in fichiers:

        if not fichier.exists():

            print()
            print("[ERREUR] Fichier introuvable :")
            print(fichier)

            sys.exit(1)

        else:

            with open(
                fichier,
                "r",
                encoding="utf-8"
            ) as f:

                lignes = [
                    ligne.strip()
                    for ligne in f
                    if ligne.strip()
                ]

            print(f"{nom} : {fichier}")
            print(f"       {len(lignes)} images")

    print()
    print("Le test ne sera PAS utilisé pendant le fine-tuning.")


# ============================================================
# MODIFICATION DE LA CONFIGURATION OFFICIELLE
# ============================================================

def creer_configuration(config_officielle):

    titre("CONFIGURATION DU FINE-TUNING")

    print("Lecture de la configuration officielle :")
    print(config_officielle)

    # --------------------------------------------------------
    # Chargement du YAML officiel
    # --------------------------------------------------------

    with open(
        config_officielle,
        "r",
        encoding="utf-8"
    ) as f:

        config = yaml.safe_load(f)

    # --------------------------------------------------------
    # Verification de l'architecture
    # --------------------------------------------------------

    if "Architecture" not in config:

        print("[ERREUR] Section 'Architecture' absente.")
        sys.exit(1)

    architecture = config["Architecture"]

    print()
    print("Architecture officielle détectée :")

    print(
        "Algorithm :",
        architecture.get("algorithm")
    )

    print(
        "Model type :",
        architecture.get("model_type")
    )

    # --------------------------------------------------------
    # IMPORTANT :
    # On conserve entièrement l'architecture officielle.
    #
    # On vérifie simplement qu'elle contient bien le Head.
    # --------------------------------------------------------

    if "Head" not in architecture:

        print()
        print("[ERREUR] La configuration officielle ne contient")
        print("pas de section Architecture -> Head.")
        print()
        print("Configuration utilisée :")
        print(config_officielle)

        sys.exit(1)

    print(
        "Head :",
        architecture["Head"].get("name")
    )

    # --------------------------------------------------------
    # GLOBAL
    # --------------------------------------------------------

    global_config = config.setdefault("Global", {})

    global_config["character_dict_path"] = str(
        DIGITS_DICT.resolve()
    )

    global_config["pretrained_model"] = str(
        PRETRAINED_FILE.resolve()
    )

    global_config["epoch_num"] = EPOCHS

    global_config["max_text_length"] = MAX_TEXT_LENGTH

    global_config["save_model_dir"] = str(
        OUTPUT_DIR.resolve()
    )

    global_config["save_res_path"] = str(
        (OUTPUT_DIR / "predicts_digits.txt").resolve()
    )

    global_config["use_gpu"] = False

    global_config["use_space_char"] = False

    global_config["model_name"] = MODEL_NAME

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = config.setdefault("Optimizer", {})

    lr_config = optimizer.setdefault("lr", {})

    lr_config["learning_rate"] = LEARNING_RATE

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_config = config.setdefault("Train", {})

    train_dataset = train_config.setdefault("dataset", {})

    train_dataset["data_dir"] = str(
        TRAIN_DATA_DIR.resolve()
    )

    train_dataset["label_file_list"] = [
        str(TRAIN_LIST.resolve())
    ]

    train_loader = train_config.setdefault("loader", {})

    train_loader["batch_size_per_card"] = BATCH_SIZE

    train_loader["num_workers"] = 0

    train_loader["shuffle"] = True

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    eval_config = config.setdefault("Eval", {})

    eval_dataset = eval_config.setdefault("dataset", {})

    eval_dataset["data_dir"] = str(
        TRAIN_DATA_DIR.resolve()
    )

    eval_dataset["label_file_list"] = [
        str(VAL_LIST.resolve())
    ]

    eval_loader = eval_config.setdefault("loader", {})

    eval_loader["batch_size_per_card"] = BATCH_SIZE

    eval_loader["num_workers"] = 0

    eval_loader["shuffle"] = False

    # --------------------------------------------------------
    # CREATION DU DOSSIER DE SORTIE
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAUVEGARDE DU YAML
    # --------------------------------------------------------

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    print()
    print("Configuration créée :")
    print(CONFIG_FILE)

    print()
    print("Architecture conservée depuis le fichier officiel.")
    print(
        "Algorithm :",
        architecture.get("algorithm")
    )

    print(
        "Head :",
        architecture["Head"].get("name")
    )


# ============================================================
# AFFICHAGE DES PARAMETRES
# ============================================================

def afficher_parametres():

    titre("PARAMETRES")

    print("Modèle              :", MODEL_NAME)
    print("Alphabet            :", ALPHABET)
    print("Nombre de caractères:", len(ALPHABET))
    print("Longueur maximale   :", MAX_TEXT_LENGTH)
    print("Epochs              :", EPOCHS)
    print("Learning rate       :", LEARNING_RATE)
    print("Batch size          :", BATCH_SIZE)
    print("Device              :", DEVICE)

    print()
    print("Train               :", TRAIN_LIST.name)
    print("Validation          :", VAL_LIST.name)
    print("Test                :", "NON utilisé pendant training")

    print()
    print("Sortie              :", OUTPUT_DIR)


# ============================================================
# LANCEMENT DU TRAINING
# ============================================================

def lancer_training():

    titre("DEBUT DU FINE-TUNING")

    print("Le modèle PP-OCRv5_server_rec va être fine-tuné.")
    print("Cela peut prendre du temps sur CPU.")
    print()

    # --------------------------------------------------------
    # Vérification finale
    # --------------------------------------------------------

    if not CONFIG_FILE.exists():

        print("[ERREUR] Configuration introuvable :")
        print(CONFIG_FILE)

        sys.exit(1)

    train_script = PADDLEOCR_DIR / "tools" / "train.py"

    if not train_script.exists():

        print("[ERREUR] Script train.py introuvable :")
        print(train_script)

        sys.exit(1)

    # --------------------------------------------------------
    # Commande
    # --------------------------------------------------------

    commande = [
        PYTHON,
        str(train_script),
        "-c",
        str(CONFIG_FILE.resolve()),
    ]

    print("Commande exécutée :")
    print(" ".join(f'"{x}"' if " " in x else x for x in commande))
    print()

    # --------------------------------------------------------
    # Exécution
    # --------------------------------------------------------

    try:

        resultat = subprocess.run(
            commande,
            cwd=PADDLEOCR_DIR,
            check=False
        )

    except KeyboardInterrupt:

        print()
        print()
        print("[INFO] Entraînement interrompu par l'utilisateur.")
        sys.exit(0)

    # --------------------------------------------------------
    # Résultat
    # --------------------------------------------------------

    if resultat.returncode != 0:

        print()
        print("[ERREUR] Le fine-tuning a échoué.")
        print("Code retour :", resultat.returncode)

        sys.exit(resultat.returncode)

    print()
    print("=" * 60)
    print("FINE-TUNING TERMINE")
    print("=" * 60)

    print()
    print("Modèle entraîné disponible dans :")
    print(OUTPUT_DIR)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    verifier_environnement()

    config_officielle = verifier_paddleocr()

    verifier_modele_preentraine()

    creer_dictionnaire()

    verifier_dataset()

    creer_configuration(
        config_officielle
    )

    afficher_parametres()

    lancer_training()


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()