import argparse
import os
import sys
from typing import Any, Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

from src.ss3_dataset import ID2LABEL
from src.ss3_model import ESM2ForSS3, freeze_module


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sequence", type=str, required=True)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    ckpt: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]
    hidden_size = ckpt["hidden_size"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    model = ESM2ForSS3(encoder=encoder, hidden_size=hidden_size, num_labels=3, dropout=0.0)
    model.head.load_state_dict(ckpt["head_state_dict"])
    freeze_module(model)

    model.to(args.device)
    model.eval()

    enc = tokenizer(
        args.sequence.strip(),
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_length,
        padding=False,
        return_tensors="pt",
    )

    enc = {k: v.to(args.device) for k, v in enc.items()}
    out = model(**enc)
    logits = out["logits"]  # [1, T, C]
    preds = logits.argmax(dim=-1)[0].tolist()  # [T]

    input_ids = enc["input_ids"][0].tolist()
    special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    non_special_positions = [i for i, is_special in enumerate(special_mask) if not is_special]

    pred_labels: List[str] = [ID2LABEL[int(preds[i])] for i in non_special_positions]
    pred_string = "".join(pred_labels)
    print(pred_string)


if __name__ == "__main__":
    main()

