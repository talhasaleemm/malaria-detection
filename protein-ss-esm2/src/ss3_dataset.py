import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset


LABEL2ID = {"H": 0, "E": 1, "C": 2}
ID2LABEL = {0: "H", 1: "E", 2: "C"}
IGNORE_INDEX = -100


def _normalize_ss3(ss3: Any) -> str:
    """
    Accept either:
      - string like "HEC..."
      - list/tuple of single-character labels
    """
    if isinstance(ss3, str):
        return ss3.strip().upper()
    if isinstance(ss3, (list, tuple)):
        return "".join(str(x).strip().upper() for x in ss3)
    raise TypeError(f"Unsupported ss3 type: {type(ss3)}")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class SS3JsonlDataset(Dataset):
    """
    Each item in JSONL must contain:
      - sequence: residue string (e.g., "ACDE...")
      - ss3: residue-aligned secondary structure string (same length as sequence)
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 1024,
    ):
        self.items = items
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.items[idx]["sequence"]
        ss3 = _normalize_ss3(self.items[idx]["ss3"])

        if len(seq) != len(ss3):
            raise ValueError(f"Alignment error: len(sequence)={len(seq)} len(ss3)={len(ss3)}")

        # Reserve room for ESM special tokens.
        # We assume `add_special_tokens=True` adds exactly 2 tokens for ESM-like tokenizers.
        max_residues = self.max_length - 2
        if len(seq) > max_residues:
            seq = seq[:max_residues]
            ss3 = ss3[:max_residues]

        enc = self.tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )  # list[bool], length == len(input_ids)
        non_special_positions = [i for i, is_special in enumerate(special_mask) if not is_special]
        if len(non_special_positions) != len(ss3):
            raise ValueError(
                f"Tokenizer alignment error: non_special_positions={len(non_special_positions)} "
                f"vs ss3_len={len(ss3)}"
            )

        labels_tokens = torch.full((len(input_ids),), IGNORE_INDEX, dtype=torch.long)
        for pos, lab_char in zip(non_special_positions, ss3):
            if lab_char not in LABEL2ID:
                raise ValueError(f"Unknown ss3 label char: {lab_char}")
            labels_tokens[pos] = LABEL2ID[lab_char]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": labels_tokens,
        }


@dataclass
class SS3Collator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_len = max(x["input_ids"].shape[0] for x in batch)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        def pad_1d(t: torch.Tensor, pad_value: int) -> torch.Tensor:
            out = torch.full((max_len,), pad_value, dtype=t.dtype)
            out[: t.shape[0]] = t
            return out

        input_ids = torch.stack([pad_1d(x["input_ids"], pad_id) for x in batch], dim=0)
        attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch], dim=0)
        labels = torch.stack([pad_1d(x["labels"], IGNORE_INDEX) for x in batch], dim=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

