# src/train_cnn.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision import models
from torch.utils.data import DataLoader

from dataset import AnimeTagDataset
from transforms import train_transform, val_transform
from evaluate import evaluate_multilabel


# ===========================
# 1. Utility to save plots
# ===========================
def save_curve(values, path, ylabel="value"):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ===========================
# 2. Evaluate function wrapper
# ===========================
def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels_np = labels.numpy()  # ground truth

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true_list.append(labels_np)
            y_prob_list.append(probs)

    metrics = evaluate_multilabel(
        y_true_list=y_true_list,
        y_prob_list=y_prob_list,
        threshold=threshold
    )
    return metrics


# ===========================
# 3. Main Training Function
# ===========================
def train_cnn(
    csv_path="labels_combined.csv",
    img_root="data/images",
    results_dir="results/cnn",
    epochs=10,
    batch_size=32,
    lr=1e-4,
    threshold=0.5,
    device=None
):
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Auto detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tag index save path
    tag_json_path = os.path.join(results_dir, "tag_to_idx.json")

    # -------------------------
    # Load datasets
    # -------------------------
    train_dataset = AnimeTagDataset(
        csv_path=csv_path,
        img_root=img_root,
        split="train",
        transform=train_transform,
        tag_json_path=tag_json_path,   # build or load tag_to_idx
    )

    val_dataset = AnimeTagDataset(
        csv_path=csv_path,
        img_root=img_root,
        split="val",
        transform=val_transform,
        tag_json_path=tag_json_path,   # ensure same tag_to_idx
        tag_to_idx=train_dataset.tag_to_idx,  # explicitly share mapping
    )

    num_tags = train_dataset.num_tags
    print(f"Number of tags (labels): {num_tags}")

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # Model setup (ResNet18)
    # -------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_tags)
    model = model.to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Training loop
    # -------------------------
    best_f1 = 0.0
    best_model_path = os.path.join(results_dir, "best_model.pt")

    train_losses = []
    val_f1_scores = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------------------------
        # Evaluate on validation set
        # -------------------------
        metrics = evaluate_model(model, val_loader, device, threshold)
        val_f1 = metrics["f1_micro"]
        val_f1_scores.append(val_f1)

        print(f"\nEpoch {epoch} — Train Loss: {avg_train_loss:.4f}, "
              f"Val F1_micro: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch}! F1_micro={val_f1:.4f}")

    # -------------------------
    # Save curves & metrics
    # -------------------------
    save_curve(train_losses, os.path.join(results_dir, "loss_curve.png"), ylabel="Train Loss")
    save_curve(val_f1_scores, os.path.join(results_dir, "f1_curve.png"), ylabel="Val F1_micro")

    final_metrics = {
        "best_f1_micro": best_f1,
        "train_losses": train_losses,
        "val_f1_scores": val_f1_scores,
        "threshold": threshold,
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\nTraining complete.")
    print(f"Best model saved to: {best_model_path}")


# ===========================
# 4. Entry point
# ===========================
if __name__ == "__main__":
    train_cnn(
        csv_path="labels_combined.csv",
        img_root="data/images",
        results_dir="results/cnn",
        epochs=10,
        batch_size=32,
        lr=1e-4,
        threshold=0.5,
    )
