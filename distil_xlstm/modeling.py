from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
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

        hidden_state = self.embedding_dropout(input_ids)
        hidden_state = self.xlstm_block_stack(input_ids)
        logits = self.lm_head(hidden_state)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        output = CausalLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_state,
        )

        return output

    @staticmethod
    def init_for_distillation(
        *,
        teacher_lm: AutoModelForCausalLM,
        xlstm_config_path: str,
        return_xlstm_config: bool = False,
    ):
        teacher_config = teacher_lm.config

        with open(xlstm_config_path, "r") as file:
            xlstm_config_dict = yaml.safe_load(file)
            xlstm_config_dict["vocab_size"] = teacher_config.vocab_size
            xlstm_config_dict["embedding_dim"] = teacher_config.hidden_size

        xlstm_config_dict = DistilxLSTMConfig.parse_xlstm_config_dict(xlstm_config_dict)
        xlstm_config = DistilxLSTMConfig(xlstm_cfg=xlstm_config_dict)
        model = DistilxLSTM(config=xlstm_config)

        if return_xlstm_config:
            return model, xlstm_config

        return model

    @staticmethod
    def init_for_distillation_with_freezed_head_and_embedding(
        *,
        teacher_lm: AutoModelForCausalLM,
        xlstm_config_path: str,
    ) -> "DistilxLSTM":
        model, config = DistilxLSTM.init_for_distillation(
            teacher_lm=teacher_lm,
            xlstm_config_path=xlstm_config_path,
            return_xlstm_config=True,
        )

        model = model.to(teacher_lm.device)

        # loading state dicts for embedding and lm_head
        model.token_embedding.load_state_dict(
            teacher_lm.model.embed_tokens.state_dict()
        )

        model.lm_head.load_state_dict(teacher_lm.lm_head.state_dict())

        if config.xlstm_cfg.tie_weights:
            model.lm_head.weight = model.token_embedding.weight

        model.token_embedding.requires_grad_(False)
        model.lm_head.requires_grad_(False)

        return model
