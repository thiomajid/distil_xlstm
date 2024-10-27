from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from xlstm import xLSTMBlockStack

from distil_xlstm.config import DistilxLSTMConfig


class DistilxLSTM(PreTrainedModel):
    config_class = DistilxLSTMConfig

    def __init__(self, config: DistilxLSTMConfig) -> None:
        super().__init__(config)

        self.config = config

        # Code from original xLSTMLMModel __init__ method
        # Same initialization has xLSTMLMModel but we use the individual components to be able
        # to use things like attention_mask coming from the tokenization step
        self.token_embedding = nn.Embedding(
            num_embeddings=config.xlstm_cfg.vocab_size,
            embedding_dim=config.xlstm_cfg.embedding_dim,
        )

        self.embedding_dropout = (
            nn.Dropout(config.xlstm_cfg.dropout)
            if config.xlstm_cfg.add_embedding_dropout
            else nn.Identity()
        )

        self.xlstm_block_stack = xLSTMBlockStack(config=config.xlstm_cfg)

        self.lm_head = nn.Linear(
            in_features=config.xlstm_cfg.embedding_dim,
            out_features=config.xlstm_cfg.vocab_size,
            bias=False,
        )

        if config.xlstm_cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        input_ids = self.token_embedding(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            input_ids = input_ids * attention_mask

        input_ids = self.embedding_dropout(input_ids)
        input_ids = self.xlstm_block_stack(input_ids)
        logits = self.lm_head(input_ids)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        output = CausalLMOutput(logits=logits, loss=loss)
        return output
