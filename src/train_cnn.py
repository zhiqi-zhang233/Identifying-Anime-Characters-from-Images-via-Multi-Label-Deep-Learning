# src/train_cnn.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import models
from torch.utils.data import DataLoader

from dataset import AnimeTagDataset
from transforms import train_transform, val_transform
from evaluate import evaluate_multilabel


# ==========================================================
# Fix filename function (remove solo_personX/)
# ==========================================================
def fix_filenames(csv_path):
    print(f"[INFO] Fixing filenames in {csv_path} ...")

    df = pd.read_csv(csv_path)
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(str(x)))

    new_csv = csv_path.replace(".csv", "_fixed.csv")
    df.to_csv(new_csv, index=False)

    print(f"[INFO] Saved fixed CSV to: {new_csv}")
    return new_csv


# ==========================================================
# Utility: save curves
# ==========================================================
def save_curve(values, path, ylabel="value"):
    plt.figure()
    plt.plot(values)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ==========================================================
# Evaluate model — handles skip batches
# ==========================================================
def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    y_true_list = []
    y_prob_list = []

    with torch.no_grad():
        for images, labels in dataloader:

            # Skip missing images
            if images == "skip":
                continue

            images = images.to(device)
            labels_np = labels.numpy()

            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true_list.append(labels_np)
            y_prob_list.append(probs)

    if len(y_true_list) == 0:
        print("[WARNING] No valid images in validation set!")
        return { "f1_micro": 0 }

    return evaluate_multilabel(y_true_list, y_prob_list, threshold=threshold)


# ==========================================================
# Main Training Function
# ==========================================================
def train_cnn(
    csv_path="../filtered_data/filtered_labels.csv",
    img_root="../images",
    results_dir="../results/cnn",
    epochs=10,
    batch_size=32,
    lr=1e-4,
    threshold=0.5,
    device=None
):
    # Fix CSV (remove folder prefix)
    csv_path = fix_filenames(csv_path)

    # Create results folder
    os.makedirs(results_dir, exist_ok=True)

    # Auto device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Save tag indices mapping
    tag_json_path = os.path.join(results_dir, "tag_to_idx.json")

    # =====================================
    # Load Datasets
    # =====================================
    train_dataset = AnimeTagDataset(
        csv_path=csv_path,
        img_root=img_root,
        split="train",
        transform=train_transform,
        tag_json_path=tag_json_path,
    )

    val_dataset = AnimeTagDataset(
        csv_path=csv_path,
        img_root=img_root,
        split="val",
        transform=val_transform,
        tag_json_path=tag_json_path,
        tag_to_idx=train_dataset.tag_to_idx,
    )

    num_tags = train_dataset.num_tags
    print(f"Number of tags: {num_tags}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # =====================================
    # Model: ResNet18
    # =====================================
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_tags)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0
    best_model_path = os.path.join(results_dir, "best_model.pt")

    train_losses = []
    val_f1_scores = []

    # =====================================
    # Training Loop
    # =====================================
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
        for batch in pbar:

            images, labels = batch

            # Skip missing images
            if images == "skip":
                continue

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)

        # =====================================
        # Validation
        # =====================================
        metrics = evaluate_model(model, val_loader, device, threshold)
        val_f1 = metrics["f1_micro"]
        val_f1_scores.append(val_f1)

        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f} | F1_micro={val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"[SAVE] New best model at epoch {epoch}: F1={val_f1:.4f}")

    # =====================================
    # Save curves & metrics
    # =====================================
    save_curve(train_losses, os.path.join(results_dir, "loss_curve.png"), ylabel="Loss")
    save_curve(val_f1_scores, os.path.join(results_dir, "f1_curve.png"), ylabel="F1_micro")

    json.dump(
        {
            "best_f1_micro": best_f1,
            "train_losses": train_losses,
            "val_f1_scores": val_f1_scores,
            "threshold": threshold,
        },
        open(os.path.join(results_dir, "metrics.json"), "w"),
        indent=2
    )

    print("\nTraining finished.")
    print(f"Best model saved to: {best_model_path}")


# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    train_cnn(
        csv_path="../filtered_data/filtered_labels.csv",
        img_root="../images",
        results_dir="../results/cnn",
        epochs=10,
        batch_size=32,
        lr=1e-4,
        threshold=0.5,
    )
