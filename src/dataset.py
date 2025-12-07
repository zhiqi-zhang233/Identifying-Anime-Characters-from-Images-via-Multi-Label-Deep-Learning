# src/dataset.py

import os
import json
from typing import List, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


def build_tag_index_from_csv(
    csv_path: str,
    min_freq: int = 1
) -> Dict[str, int]:
    """
    Build a tag -> index dictionary from the tags column in labels.csv.

    Args:
        csv_path: Path to labels.csv.
        min_freq: Only keep tags whose frequency >= min_freq.
                  This helps you filter out very rare tags if needed.

    Returns:
        tag_to_idx: A dictionary mapping each tag string to an integer index.
    """
    df = pd.read_csv(csv_path)

    # The 'tags' column contains space-separated tag strings
    all_tags = {}
    for tags_str in df["tags"]:
        if isinstance(tags_str, str):
            tags = tags_str.split()
            for t in tags:
                all_tags[t] = all_tags.get(t, 0) + 1

    # Filter by minimum frequency
    filtered_tags = [t for t, c in all_tags.items() if c >= min_freq]

    # Sort tags for reproducibility (so everyone has the same order)
    filtered_tags.sort()

    tag_to_idx = {tag: idx for idx, tag in enumerate(filtered_tags)}
    return tag_to_idx


def save_tag_index(
    tag_to_idx: Dict[str, int],
    json_path: str
) -> None:
    """
    Save tag_to_idx dictionary to a JSON file.

    Args:
        tag_to_idx: Mapping from tag string to index.
        json_path: Path where JSON will be saved.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tag_to_idx, f, ensure_ascii=False, indent=2)


def load_tag_index(
    json_path: str
) -> Dict[str, int]:
    """
    Load tag_to_idx dictionary from a JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        tag_to_idx: Mapping from tag string to index.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        tag_to_idx = json.load(f)
    # JSON keys are strings already, values are ints
    return tag_to_idx


def split_dataframe(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the full DataFrame into train/val/test by row indices.

    Args:
        df: Full DataFrame of all samples.
        val_ratio: Proportion of samples used for validation.
        test_ratio: Proportion of samples used for test.
        random_seed: Seed for reproducible shuffling.

    Returns:
        train_df, val_df, test_df: Three DataFrames representing each split.
    """
    assert 0 <= val_ratio < 1
    assert 0 <= test_ratio < 1
    assert val_ratio + test_ratio < 1

    # Total number of samples
    n = len(df)

    # Shuffle indices to randomize the split
    rng = df.sample(frac=1.0, random_state=random_seed).index.to_list()

    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test

    train_idx = rng[:n_train]
    val_idx = rng[n_train:n_train + n_val]
    test_idx = rng[n_train + n_val:]

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


class AnimeTagDataset(Dataset):
    """
    PyTorch Dataset for anime images with multi-label tags.

    It reads from labels.csv, loads images from disk, and converts the tag
    string into a multi-hot label vector using a shared tag_to_idx mapping.

    Each sample is:
        image: FloatTensor [C, H, W]
        labels: FloatTensor [num_tags] (multi-hot, 0 or 1)
    """

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
            csv_path: Path to labels.csv.
            img_root: Directory where all images are stored.
                      Example: data/images/
            split: "train", "val", or "test".
            transform: Torchvision transform to apply to PIL image.
            tag_to_idx: Pre-built tag_to_idx mapping. If None, it will be
                        built from csv_path.
            tag_json_path: Optional path to save or load tag_to_idx.
                           If provided and file exists, we load it.
                           If not exist yet, we build from csv and then save.
            min_tag_freq: Minimum frequency for tags when building tag_to_idx.
            val_ratio, test_ratio, random_seed: Controls DataFrame splitting.
        """
        super().__init__()

        self.csv_path = csv_path
        self.img_root = img_root
        self.transform = transform

        # 1. Load full CSV
        full_df = pd.read_csv(csv_path)

        # 2. Build or load tag_to_idx
        if tag_to_idx is not None:
            # Use the mapping passed in from outside
            self.tag_to_idx = tag_to_idx
        else:
            # If a JSON path is provided and exists, load from it
            if tag_json_path is not None and os.path.exists(tag_json_path):
                self.tag_to_idx = load_tag_index(tag_json_path)
            else:
                # Otherwise build mapping from CSV, then optionally save it
                self.tag_to_idx = build_tag_index_from_csv(
                    csv_path,
                    min_freq=min_tag_freq
                )
                if tag_json_path is not None:
                    save_tag_index(self.tag_to_idx, tag_json_path)

        self.num_tags = len(self.tag_to_idx)

        # 3. Split DataFrame into train/val/test consistently
        train_df, val_df, test_df = split_dataframe(
            full_df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.df)

    def _encode_tags(self, tags_str: str) -> torch.Tensor:
        """
        Convert a space-separated tag string into a multi-hot vector.

        Args:
            tags_str: e.g. "1girl pink_hair long_hair smile"

        Returns:
            labels: FloatTensor of shape [num_tags],
                    each element is 0 or 1.
        """
        labels = torch.zeros(self.num_tags, dtype=torch.float32)

        if not isinstance(tags_str, str):
            # In case tags_str is NaN or missing
            return labels

        tags = tags_str.split()
        for t in tags:
            if t in self.tag_to_idx:
                idx = self.tag_to_idx[t]
                labels[idx] = 1.0

        return labels

    def __getitem__(self, index: int):
        """
        Get a single sample given its index.

        Returns:
            image: FloatTensor [C, H, W]
            labels: FloatTensor [num_tags]
        """
        row = self.df.iloc[index]

        # The filename column contains relative path, e.g. "solo_person1/9999993.jpg"
        filename = row["filename"]
        img_path = os.path.join(self.img_root, filename)

        # Load the image with PIL
        image = Image.open(img_path).convert("RGB")

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Encode tags to multi-hot vector
        tags_str = row["tags"]
        labels = self._encode_tags(tags_str)

        return image, labels
