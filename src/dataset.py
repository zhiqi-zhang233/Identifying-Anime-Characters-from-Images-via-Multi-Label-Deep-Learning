# src/dataset.py

import os
import json
from typing import Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


# ==========================================================
# Build tag index
# ==========================================================
def build_tag_index_from_csv(csv_path: str, min_freq: int = 1) -> Dict[str, int]:
    df = pd.read_csv(csv_path)

    tag_freq = {}
    for tags_str in df["tags"]:
        if isinstance(tags_str, str):
            for tag in tags_str.split():
                tag_freq[tag] = tag_freq.get(tag, 0) + 1

    # keep tags with freq >= min_freq
    tags = [t for t, c in tag_freq.items() if c >= min_freq]
    tags.sort()

    return {tag: idx for idx, tag in enumerate(tags)}


def save_tag_index(tag_to_idx: Dict[str, int], json_path: str):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tag_to_idx, f, ensure_ascii=False, indent=2)


def load_tag_index(json_path: str) -> Dict[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================================
# Split dataframe
# ==========================================================
def split_dataframe(df: pd.DataFrame, val_ratio: float = 0.1, test_ratio: float = 0.1, random_seed: int = 42):
    assert val_ratio + test_ratio < 1

    N = len(df)
    shuffled_idx = df.sample(frac=1.0, random_state=random_seed).index.tolist()

    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)
    n_train = N - n_val - n_test

    train_df = df.loc[shuffled_idx[:n_train]].reset_index(drop=True)
    val_df = df.loc[shuffled_idx[n_train:n_train+n_val]].reset_index(drop=True)
    test_df = df.loc[shuffled_idx[n_train+n_val:]].reset_index(drop=True)

    return train_df, val_df, test_df


# ==========================================================
# Dataset class
# ==========================================================
class AnimeTagDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        split: str = "train",
        transform=None,
        tag_to_idx: Optional[Dict[str, int]] = None,
        tag_json_path: Optional[str] = None,
        min_tag_freq: int = 1,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Args:
            csv_path     : filtered_labels_fixed.csv
            img_root     : images folder
            split        : train / val / test
            transform    : transform function
            tag_to_idx   : pass train mapping to val/test for consistency
            tag_json_path: save/load json for reproducibility
        """
        self.csv_path = csv_path
        self.img_root = img_root
        self.transform = transform

        # ------------------------------------------------------
        # Load full CSV
        # ------------------------------------------------------
        full_df = pd.read_csv(csv_path)

        # ------------------------------------------------------
        # Build or load tag_to_idx
        # ------------------------------------------------------
        if tag_to_idx is not None:
            self.tag_to_idx = tag_to_idx

        else:
            if tag_json_path is not None and os.path.exists(tag_json_path):
                self.tag_to_idx = load_tag_index(tag_json_path)
            else:
                self.tag_to_idx = build_tag_index_from_csv(csv_path, min_freq=min_tag_freq)
                if tag_json_path is not None:
                    save_tag_index(self.tag_to_idx, tag_json_path)

        self.num_tags = len(self.tag_to_idx)

        # ------------------------------------------------------
        # Data split
        # ------------------------------------------------------
        train_df, val_df, test_df = split_dataframe(
            full_df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )

        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.df)

    # ------------------------------------------------------
    # Encode tags → multi-hot vector
    # ------------------------------------------------------
    def _encode_tags(self, tags_str: str) -> torch.Tensor:
        labels = torch.zeros(self.num_tags, dtype=torch.float32)

        if isinstance(tags_str, str):
            for t in tags_str.split():
                if t in self.tag_to_idx:
                    labels[self.tag_to_idx[t]] = 1.0

        return labels

    # ------------------------------------------------------
    # IMPORTANT: Safe image loading (skip missing images)
    # ------------------------------------------------------
    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        filename = row["filename"]
        img_path = os.path.join(self.img_root, filename)

        # --------------------------
        # TRY TO LOAD IMAGE
        # --------------------------
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # → train loop will skip batches containing "skip"
            return "skip", "skip"

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        labels = self._encode_tags(row["tags"])

        return image, labels
