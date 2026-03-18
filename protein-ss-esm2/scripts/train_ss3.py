import argparse
import json
import os
from typing import Any, Dict

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import sys

# Allow running scripts directly (no package install required).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ss3_dataset import SS3JsonlDataset, SS3Collator, read_jsonl
from src.ss3_model import ESM2ForSS3, freeze_module
from src.metrics import token_metrics_from_logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="runs/esm2_ss3")
    p.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--freeze_encoder", action="store_true", default=True)
    p.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16")

    mixed_precision = "fp16" if args.fp16 else ("bf16" if args.bf16 else "no")
    accelerator = Accelerator(mixed_precision=mixed_precision)
    if accelerator.is_main_process:
        print(f"[info] Using mixed_precision={mixed_precision}")

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    hidden_size = encoder.config.hidden_size

    model = ESM2ForSS3(encoder=encoder, hidden_size=hidden_size, num_labels=3, dropout=0.1)
    if args.freeze_encoder:
        freeze_module(model.encoder)

    train_items = read_jsonl(args.train_jsonl)
    val_items = read_jsonl(args.val_jsonl)

    train_ds = SS3JsonlDataset(train_items, tokenizer=tokenizer, max_length=args.max_length)
    val_ds = SS3JsonlDataset(val_items, tokenizer=tokenizer, max_length=args.max_length)

    collator = SS3Collator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Simple cosine schedule would be fine; keep it minimal for clarity.
    from transformers import get_linear_schedule_with_warmup

    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_macro_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(args.epochs):
        model.train()
        if accelerator.is_main_process:
            print(f"[train] epoch {epoch+1}/{args.epochs}")

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
        for batch in pbar:
            outputs = model(**batch)
            loss = outputs["loss"]

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        # ---- Validation ----
        model.eval()
        all_token_acc = []
        all_macro_f1 = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                metrics = token_metrics_from_logits(
                    logits=outputs["logits"],
                    labels=batch["labels"],
                    id2label={0: "H", 1: "E", 2: "C"},
                )
                all_token_acc.append(metrics["token_acc"])
                all_macro_f1.append(metrics["macro_f1"])

        val_token_acc = float(sum(all_token_acc) / max(1, len(all_token_acc)))
        val_macro_f1 = float(sum(all_macro_f1) / max(1, len(all_macro_f1)))
        accelerator.print(f"[val] token_acc={val_token_acc:.4f} macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                ckpt: Dict[str, Any] = {
                    "model_name": args.model_name,
                    "hidden_size": hidden_size,
                    "num_labels": 3,
                    "labels": ["H", "E", "C"],
                    "head_state_dict": unwrapped.head.state_dict(),
                }
                torch.save(ckpt, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {"best_macro_f1": best_macro_f1, "val_token_acc": val_token_acc},
                        f,
                        indent=2,
                    )

    if accelerator.is_main_process:
        print(f"[done] best_macro_f1={best_macro_f1:.4f} saved to {best_path}")


if __name__ == "__main__":
    main()

