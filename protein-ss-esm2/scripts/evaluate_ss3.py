import argparse
import json
import os
import sys
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.ss3_dataset import SS3JsonlDataset, SS3Collator, read_jsonl, ID2LABEL, IGNORE_INDEX
from src.ss3_model import ESM2ForSS3
from src.metrics import token_metrics_from_logits

from sklearn.metrics import confusion_matrix


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, required=True)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    ckpt: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)

    hidden_size = ckpt["hidden_size"]
    model = ESM2ForSS3(encoder=encoder, hidden_size=hidden_size, num_labels=3, dropout=0.1)
    model.head.load_state_dict(ckpt["head_state_dict"])

    model.eval()

    test_items = read_jsonl(args.test_jsonl)
    ds = SS3JsonlDataset(test_items, tokenizer=tokenizer, max_length=args.max_length)
    collator = SS3Collator(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    all_token_acc = []
    all_macro_f1 = []
    all_y_true = []
    all_y_pred = []

    for batch in loader:
        outputs = model(**batch)
        metrics = token_metrics_from_logits(
            logits=outputs["logits"],
            labels=batch["labels"],
            id2label=ID2LABEL,
        )
        all_token_acc.append(metrics["token_acc"])
        all_macro_f1.append(metrics["macro_f1"])

        preds = outputs["logits"].argmax(dim=-1)  # [B, T]
        labels = batch["labels"]                 # [B, T]
        mask = labels != IGNORE_INDEX
        if mask.any():
            all_y_true.append(labels[mask].detach().cpu())
            all_y_pred.append(preds[mask].detach().cpu())

    token_acc = float(sum(all_token_acc) / max(1, len(all_token_acc)))
    macro_f1 = float(sum(all_macro_f1) / max(1, len(all_macro_f1)))

    if all_y_true:
        y_true = torch.cat(all_y_true).numpy()
        y_pred = torch.cat(all_y_pred).numpy()
        labels_order = sorted(ID2LABEL.keys())
        names = [ID2LABEL[i] for i in labels_order]
        cm = confusion_matrix(y_true, y_pred, labels=labels_order)
        print("Confusion matrix (all test batches):")
        print(names)
        print(cm)

    results = {"token_acc": token_acc, "macro_f1": macro_f1}
    out_path = os.path.join(os.path.dirname(args.checkpoint), "test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[test] token_acc={token_acc:.4f} macro_f1={macro_f1:.4f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

