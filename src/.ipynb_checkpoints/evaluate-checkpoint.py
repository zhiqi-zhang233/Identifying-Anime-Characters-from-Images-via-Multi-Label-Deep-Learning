# src/evaluate.py

from typing import List, Dict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_multilabel(
    y_true_list: List[np.ndarray],
    y_prob_list: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate multi-label predictions using micro/macro F1, precision, recall.

    Args:
        y_true_list: A list of numpy arrays, each with shape [batch_size, num_tags],
                     representing ground-truth multi-hot labels.
        y_prob_list: A list of numpy arrays, each with shape [batch_size, num_tags],
                     representing predicted probabilities in [0, 1].
        threshold: Probability threshold to convert to binary predictions.

    Returns:
        metrics: A dictionary containing micro/macro F1, precision, recall, e.g.
                 {
                   "f1_micro": 0.75,
                   "f1_macro": 0.68,
                   "precision_micro": ...,
                   "recall_micro": ...,
                   ...
                 }
    """
    # Concatenate all batches along axis 0
    y_true = np.concatenate(y_true_list, axis=0)  # [N, num_tags]
    y_prob = np.concatenate(y_prob_list, axis=0)  # [N, num_tags]

    # Convert probabilities to 0/1 predictions using the given threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Safety check: ensure int type
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Micro-averaged scores: treat all (sample, tag) pairs equally
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    precision_micro = precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    recall_micro = recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )

    # Macro-averaged scores: compute metrics per tag, then average across tags
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "f1_micro": f1_micro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "threshold": threshold,
    }

    return metrics


def search_best_threshold(
    y_true_list: List[np.ndarray],
    y_prob_list: List[np.ndarray],
    thresholds: List[float]
) -> Dict[str, float]:
    """
    Simple grid search over thresholds to find the best micro-F1.

    Args:
        y_true_list: List of ground-truth arrays.
        y_prob_list: List of predicted probability arrays.
        thresholds: A list of thresholds to try, e.g. [0.3, 0.4, ..., 0.7].

    Returns:
        best_result: A dictionary with the best threshold and metrics.
    """
    best_f1 = -1.0
    best_result = {}

    for thr in thresholds:
        metrics = evaluate_multilabel(y_true_list, y_prob_list, threshold=thr)
        if metrics["f1_micro"] > best_f1:
            best_f1 = metrics["f1_micro"]
            best_result = metrics

    return best_result
