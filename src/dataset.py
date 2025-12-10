# src/dataset.py

import os
import json
from typing import Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


# ==========================================================
# Build tag index
# ==========================================================
def build_tag_index_from_csv(csv_path: str, min_freq: int = 1) -> Dict[str, int]:
    """
    Build tag → index dictionary from CSV.
    """
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
def split_dataframe(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Random shuffle + split into train/val/test
    """
    assert val_ratio + test_ratio < 1

    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    N = len(df)

    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)

    train_df = df.iloc[:-n_val - n_test]
    val_df = df.iloc[-n_val - n_test:-n_test]
    test_df = df.iloc[-n_test:]

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ==========================================================
# Dataset class
# ==========================================================
class AnimeTagDataset(Dataset):
    """
    Dataset that loads anime character images + multi-label tags.
    With safe image loading (missing images → skip).
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        split="train",
        transform=None,
        tag_to_idx=None,
        tag_json_path=None,
        min_tag_freq=1,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    ):
        self.csv_path = csv_path
        self.img_root = img_root
        self.transform = transform

        # ------------------------------------------------------
        # Load CSV
        # ------------------------------------------------------
        df = pd.read_csv(csv_path)

        # ------------------------------------------------------
        # tag_to_idx mapping
        # ------------------------------------------------------
        if tag_to_idx is not None:
            # val/test reuse mapping from train
            self.tag_to_idx = tag_to_idx
        else:
            if tag_json_path and os.path.exists(tag_json_path):
                self.tag_to_idx = load_tag_index(tag_json_path)
            else:
                self.tag_to_idx = build_tag_index_from_csv(csv_path, min_freq=min_tag_freq)
                if tag_json_path:
                    save_tag_index(self.tag_to_idx, tag_json_path)

        self.num_tags = len(self.tag_to_idx)

        # ------------------------------------------------------
        # Split the dataframe
        # ------------------------------------------------------
        train_df, val_df, test_df = split_dataframe(df, val_ratio, test_ratio, random_seed)

        if split == "train":
            self.df = train_df
        elif split == "val":
            self.df = val_df
        elif split == "test":
            self.df = test_df
        else:
            raise ValueError("split must be train/val/test")

    # ------------------------------------------------------
    # Encode a tag string into a multi-hot vector
    # ------------------------------------------------------
    def _encode_tags(self, tags_str: str):
        labels = torch.zeros(self.num_tags, dtype=torch.float32)

        if isinstance(tags_str, str):
            for t in tags_str.split():
                if t in self.tag_to_idx:
                    labels[self.tag_to_idx[t]] = 1.0
        return labels

    # ------------------------------------------------------
    # Safe __getitem__
    # ------------------------------------------------------
    def __getitem__(self, index: int):
        """
        Returns:
            (image_tensor, labels_tensor)
        Or if image missing:
            ("skip", labels)
        """
        row = self.df.iloc[index]

        # 原始文件名，例如 "solo_person3/5999590.jpg"
        orig_name = row["filename"]

        # 拆出子目录名和纯文件名
        parts = orig_name.split("/")
        if len(parts) > 1:
            dir_name = parts[0]            # solo_person3
            base_name = parts[-1]          # 5999590.jpg
        else:
            dir_name = ""
            base_name = orig_name

        # 构造一系列“候选路径”，按顺序逐个尝试
        candidates = []

        # 1) data/images/solo_person3/5999590.jpg  （严格按 CSV）
        candidates.append(os.path.join(self.img_root, orig_name))

        # 2) data/images/solo_person3/_5999590.jpg
        if dir_name:
            candidates.append(os.path.join(self.img_root, dir_name, "_" + base_name))

        # 3) data/images/5999590.jpg  （扁平目录，不带下划线）
        candidates.append(os.path.join(self.img_root, base_name))

        # 4) data/images/_5999590.jpg （扁平目录，带下划线）
        candidates.append(os.path.join(self.img_root, "_" + base_name))

        # 5) data/images/_solo_person3/5999590.jpg （以防有人那样命名）
        candidates.append(os.path.join(self.img_root, "_" + orig_name))

        img_path = None
        for p in candidates:
            if os.path.exists(p):
                img_path = p
                break

        labels = self._encode_tags(row["tags"])

        # 所有候选路径都不存在，就跳过这个样本
        if img_path is None:
            return "skip", labels

        # 尝试读取图片
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return "skip", labels

        if self.transform:
            image = self.transform(image)

        return image, labels



    def __len__(self):
        return len(self.df)
