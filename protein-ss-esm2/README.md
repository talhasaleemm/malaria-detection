# Protein Secondary Structure with ESM-2 (Residue-Level Q3)

This project fine-tunes a lightweight ESM-2 checkpoint (via Hugging Face) to predict **secondary structure per residue** using a token-level classification head.

## What you can train
- **Q3 secondary structure**: `H` (helix), `E` (strand), `C` (coil)
- Objective: predict a label for each amino-acid residue (special tokens are ignored in loss)

## Model
- Encoder: `facebook/esm2_t6_8M_UR50D` by default
- Head: `Linear(hidden_size -> 3)` applied to `last_hidden_state` at each token position
- Default training approach: **freeze encoder**, train only the head (fast + portfolio-friendly)

## Data format (JSONL)
Provide line-delimited JSON with this schema:

```json
{"sequence": "ACDEFGHIK...", "ss3": "HECCHH..."}
```

Requirements:
- `len(sequence) == len(ss3)` (both are residue-aligned)
- Allowed `ss3` characters: `H`, `E`, `C` (case-insensitive)

## Quickstart
1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
   (Make sure you are using Python 3.)
2. Train:
   ```bash
   python scripts/train_ss3.py ^
     --train_jsonl data/train.jsonl ^
     --val_jsonl data/val.jsonl ^
     --output_dir runs/esm2_ss3
   ```
3. Evaluate:
   ```bash
   python scripts/evaluate_ss3.py ^
     --checkpoint runs/esm2_ss3/best.pt ^
     --test_jsonl data/test.jsonl
   ```
4. Infer:
   ```bash
   python scripts/infer_ss3.py ^
     --checkpoint runs/esm2_ss3/best.pt ^
     --sequence "MKT..."
   ```

## Notes
- ESM tokenizers add special tokens; this project aligns labels onto **non-special** token positions and uses `-100` for ignored positions in loss.
- For very long proteins, the script truncates to `--max_length` (default `1024`).

