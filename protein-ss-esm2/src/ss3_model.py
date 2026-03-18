from typing import Optional

import torch
import torch.nn as nn

from .ss3_dataset import IGNORE_INDEX


class SS3Head(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: [B, T, H]
        x = self.dropout(last_hidden_state)
        return self.classifier(x)  # [B, T, C]


class ESM2ForSS3(nn.Module):
    """
    Wraps an ESM-2 encoder and a per-token classification head.
    """

    def __init__(self, encoder, hidden_size: int, num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = SS3Head(hidden_size=hidden_size, num_labels=num_labels, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(out.last_hidden_state)  # [B, T, C]

        if labels is None:
            return {"logits": logits}

        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),  # [B*T, C]
            labels.view(-1),                    # [B*T]
        )
        return {"loss": loss, "logits": logits}


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False

