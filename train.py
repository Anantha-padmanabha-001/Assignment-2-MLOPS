import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import matplotlib.pyplot as plt

from data_prep import prepare_dataset

# ------------------------------------------------------------
# 1. Preprocess data (M1.1)
# ------------------------------------------------------------
if not os.path.exists("data/processed/train"):
    print("Preprocessing data...")
    prepare_dataset(raw_dir="data/raw", out_dir="data/processed")
else:
    print("data/processed already exists, skipping preprocessing.")

# ------------------------------------------------------------
# 2. Build model (M1.2)
# ------------------------------------------------------------
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 28 * 28, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def manual_confusion_matrix(labels, preds, num_classes=2):
    """2x2 confusion matrix without sklearn/scipy (avoids the DLL block)."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    for actual, pred in zip(labels, preds):
        cm[actual][pred] += 1
    return cm

train_ds = datasets.ImageFolder("data/processed/train", transform=TRANSFORM)
val_ds = datasets.ImageFolder("data/processed/val", transform=TRANSFORM)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model = SimpleCNN().to(device)

# ------------------------------------------------------------
# 3. Train with MLflow tracking (M1.3)
# ------------------------------------------------------------
mlflow.set_experiment("catsdogs-classification")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
EPOCHS = 5

with mlflow.start_run():
    mlflow.log_params({"epochs": EPOCHS, "batch_size": 32, "lr": 1e-3, "model": "SimpleCNN"})
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
        print(f"epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.savefig("loss_curve.png")
    mlflow.log_artifact("loss_curve.png")

    cm = manual_confusion_matrix(all_labels, all_preds, num_classes=2)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (val)")
    plt.colorbar()
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    torch.save(model.state_dict(), "catsdogs_model.pt")
    mlflow.log_artifact("catsdogs_model.pt")

print("\nTraining complete. Model saved as catsdogs_model.pt")