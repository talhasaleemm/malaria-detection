from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .ss3_dataset import IGNORE_INDEX


def token_metrics_from_logits(
    logits,  # torch.Tensor [B, T, C]
    labels,  # torch.Tensor [B, T] with IGNORE_INDEX
    id2label: Dict[int, str],
) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)  # [B, T]

    mask = labels != IGNORE_INDEX
    if mask.sum().item() == 0:
        return {"token_acc": 0.0, "macro_f1": 0.0}

    y_true = labels[mask].detach().cpu().numpy().tolist()
    y_pred = preds[mask].detach().cpu().numpy().tolist()

    labels_order = sorted(id2label.keys())
    token_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels_order, average="macro")

    return {"token_acc": float(token_acc), "macro_f1": float(macro_f1)}


@torch.no_grad()
def confusion_matrix_from_logits(
    logits,  # torch.Tensor [B, T, C]
    labels,  # torch.Tensor [B, T]
    id2label: Dict[int, str],
) -> Tuple[np.ndarray, List[str]]:
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX

    y_true = labels[mask].detach().cpu().numpy()
    y_pred = preds[mask].detach().cpu().numpy()

    labels_order = sorted(id2label.keys())
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    names = [id2label[i] for i in labels_order]
    return cm, names

