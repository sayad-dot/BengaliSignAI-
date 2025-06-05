# scripts/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from src.python.dataset import SignDataset
from src.python.model import Simple3DCNN
from tqdm import tqdm

PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../models"
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15

def get_class_list(processed_dir):
    return sorted(
        [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    )

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    classes = get_class_list(PROCESSED_DIR)
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    full_dataset = SignDataset(PROCESSED_DIR, classes_list=classes)
    total_samples = len(full_dataset)
    val_size = int(total_samples * VAL_SPLIT)
    train_size = total_samples - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Simple3DCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / train_size
        train_acc = correct_train / total_train

        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc = correct_val / total_val
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODELS_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": classes
            }, save_path)
            print(f"  Saved new best model (Val Acc: {val_acc:.4f})\n")

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()
