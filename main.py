import os
import random
import glob
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# -------------------------------------------------------------------------
# 1. Configuration & Dictionnaire de caractères
# -------------------------------------------------------------------------
DATA_DIR = "Data"
TRAIN_COUNT = 10
TEST_COUNT = 15
IMG_HEIGHT = 32
IMG_WIDTH = 100
BATCH_SIZE = 4
EPOCHS = 35
LR = 0.001

# Chiffres et séparateurs fréquents sur cartes marines
CHARS = "0123456789.,"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 est réservé au blank CTC
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}

# -------------------------------------------------------------------------
# 2. Dataset et Chargement des images
# -------------------------------------------------------------------------
class MarineDigitDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)

        # Extraction du label depuis le nom du fichier (ex: "14.2_01.png" -> "14.2")
        basename = os.path.splitext(os.path.basename(path))[0]
        raw_label = basename.split("_")[0]  # Adaptez selon votre convention
        
        # Filtre pour ne garder que les caractères autorisés
        clean_label = "".join([c for c in raw_label if c in CHAR2IDX])
        label_tensor = torch.tensor([CHAR2IDX[c] for c in clean_label], dtype=torch.long)

        return img, label_tensor, clean_label

def collate_fn(batch):
    imgs, labels, raw_labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    label_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    targets = torch.cat(labels)
    return imgs, targets, label_lengths, raw_labels

# -------------------------------------------------------------------------
# 3. Architecture CRNN (CNN + BiLSTM + CTC)
# -------------------------------------------------------------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16 x 50
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8 x 25
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((8, 1)) # 1 x 25
        )
        self.rnn = nn.LSTM(128, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.cnn(x)              # [B, 128, 1, W_seq]
        features = features.squeeze(2)       # [B, 128, W_seq]
        features = features.permute(0, 2, 1) # [B, W_seq, 128]
        rnn_out, _ = self.rnn(features)      # [B, W_seq, 128]
        out = self.fc(rnn_out)               # [B, W_seq, num_classes]
        return out.log_softmax(2)

def decode_prediction(pred_indices):
    """Décode les indices CTC en chaîne de texte en supprimant répétitions et blanks."""
    chars = []
    prev = 0
    for idx in pred_indices:
        if idx != 0 and idx != prev:
            chars.append(IDX2CHAR.get(idx, ""))
        prev = idx
    return "".join(chars)

# -------------------------------------------------------------------------
# 4. Préparation des données & Entraînement
# -------------------------------------------------------------------------
all_files = glob.glob(os.path.join(DATA_DIR, "*.*"))
valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts]

if len(image_files) < (TRAIN_COUNT + TEST_COUNT):
    raise ValueError(f"Le dossier '{DATA_DIR}' contient {len(image_files)} images, il en faut au moins {TRAIN_COUNT + TEST_COUNT}.")

random.seed(42)
random.shuffle(image_files)

train_files = image_files[:TRAIN_COUNT]
test_files = image_files[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

print(f"Échantillons d'entraînement : {len(train_files)}")
print(f"Échantillons de test : {len(test_files)}")

train_loader = DataLoader(MarineDigitDataset(train_files), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(MarineDigitDataset(test_files), batch_size=1, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(num_classes=len(CHARS) + 1).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
test_accuracies = []

print("\n--- Début de l'entraînement ---")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, targets, target_lengths, _ in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        preds = model(imgs) # [B, T, C]
        preds = preds.permute(1, 0, 2) # CTC demande [T, B, C]

        input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(0), dtype=torch.long)
        loss = criterion(preds, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Évaluation sur le test
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, _, _, raw_labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs) # [1, T, C]
            pred_idx = preds.argmax(2).squeeze(0).tolist()
            pred_str = decode_prediction(pred_idx)
            if pred_str == raw_labels[0]:
                correct += 1
    acc = (correct / TEST_COUNT) * 100
    test_accuracies.append(acc)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Précision Test: {acc:.1f}%")

# -------------------------------------------------------------------------
# 5. Affichage des courbes et d'exemples de test
# -------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))

# Graphique de la Loss
plt.subplot(2, 2, 1)
plt.plot(range(1, EPOCHS + 1), train_losses, color="royalblue", lw=2)
plt.title("Perte d'entraînement (CTC Loss)")
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# Graphique de la Précision
plt.subplot(2, 2, 2)
plt.plot(range(1, EPOCHS + 1), test_accuracies, color="seagreen", lw=2)
plt.title("Précision sur le jeu de test (16 images)")
plt.xlabel("Époques")
plt.ylabel("Exactitude (%)")
plt.grid(True, alpha=0.3)

# Affichage de 4 vignettes de test avec leurs prédictions
model.eval()
sample_test_loader = DataLoader(MarineDigitDataset(test_files[:4]), batch_size=1, shuffle=False, collate_fn=collate_fn)
for i, (imgs, _, _, raw_labels) in enumerate(sample_test_loader):
    with torch.no_grad():
        preds = model(imgs.to(device))
        pred_idx = preds.argmax(2).squeeze(0).tolist()
        pred_str = decode_prediction(pred_idx)

    ax = plt.subplot(2, 4, 5 + i)
    img_disp = imgs.squeeze().cpu().numpy() * 0.5 + 0.5
    ax.imshow(img_disp, cmap="gray")
    ax.set_title(f"Vrai: '{raw_labels[0]}'\nPrédit: '{pred_str}'", color="green" if raw_labels[0] == pred_str else "red")
    ax.axis("off")

plt.tight_layout()
plt.show()