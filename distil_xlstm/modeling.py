import copy
from pathlib import Path
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from xlstm import xLSTMBlockStack

from distil_xlstm.config import DistilxLSTMConfig
from distil_xlstm.optim.loss import FrobeniusNormComputation
from distil_xlstm.utils import (
    count_parameters,
    count_trainable_parameters,
    xLSTMCausalLMOutput,
)


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
            # padding_idx=config.pad_token_id,
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
        frobenius_computation: Optional[FrobeniusNormComputation] = None,
        **kwargs,
    ):
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        hidden_states_per_block: torch.Tensor | None = None
        if frobenius_computation == "ratio":
            hidden_states_per_block = torch.empty(
                size=(
                    self.config.xlstm_cfg.num_blocks,
                    hidden_states.size(0),
                    hidden_states.size(2),
                )
            )

            for idx, block in enumerate(self.xlstm_block_stack.blocks):
                hidden_states_per_block[idx] = block(hidden_states)

            last_hidden_state = self.xlstm_block_stack.post_blocks_norm(
                hidden_states_per_block[-1]
            )

            hidden_states = last_hidden_state

        hidden_states = self.xlstm_block_stack(hidden_states)
        logits: torch.Tensor = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
            shift_logits = rearrange(
                logits[..., :-1, :].contiguous(), "b s v -> (b s) v"
            )

            # shape: [batch, seq] -> [batch * (seq-1)]
            shift_labels = rearrange(labels[..., 1:].contiguous(), "b s -> (b s)")

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                input=shift_logits,
                target=shift_labels,
                # ignore_index=self.config.pad_token_id,
            )

        return xLSTMCausalLMOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            hidden_states_per_block=hidden_states_per_block,
        )

    @staticmethod
    def init_for_distillation(
        *,
        teacher_config: AutoConfig,
        tokenizer: AutoTokenizer,
        xlstm_config_path: str,
        return_xlstm_config: bool = False,
        v2: bool = True,
    ):
        with open(xlstm_config_path, "r") as file:
            xlstm_config_dict = yaml.safe_load(file)
            xlstm_config_dict["vocab_size"] = teacher_config.vocab_size
            xlstm_config_dict["embedding_dim"] = teacher_config.hidden_size

        if v2:
            num_blocks = teacher_config.num_hidden_layers // 2
            xlstm_config_dict["num_blocks"] = num_blocks
            xlstm_config_dict["slstm_at"] = list(range(0, num_blocks - 1, 2))

            teacher_num_heads = teacher_config.num_attention_heads
            while teacher_num_heads % 4 != 0:
                teacher_num_heads += 1

            xlstm_config_dict["mlstm_block"]["mlstm"]["num_heads"] = teacher_num_heads
            xlstm_config_dict["slstm_block"]["slstm"]["num_heads"] = teacher_num_heads

        parsed_config = DistilxLSTMConfig.parse_xlstm_config_dict(
            copy.deepcopy(xlstm_config_dict)
        )

        xlstm_config = DistilxLSTMConfig(xlstm_cfg=parsed_config)
        xlstm_config.pad_token_id = tokenizer.pad_token_id

        model = DistilxLSTM(config=xlstm_config)

        if return_xlstm_config:
            return model, xlstm_config

        return model

    @staticmethod
    def init_for_distillation_with_freezed_head_and_embedding(
        *,
        teacher_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        xlstm_config_path: str,
        v2: bool = True,
    ) -> "DistilxLSTM":
        model, config = DistilxLSTM.init_for_distillation(
            teacher_config=teacher_model.config,
            tokenizer=tokenizer,
            xlstm_config_path=xlstm_config_path,
            return_xlstm_config=True,
            v2=v2,
        )

        model = model.to(teacher_model.device)
        model.token_embedding.load_state_dict(
            teacher_model.model.embed_tokens.state_dict()
        )
        model.lm_head.load_state_dict(teacher_model.lm_head.state_dict())

        if config.xlstm_cfg.tie_weights:
            model.lm_head.weight = model.token_embedding.weight

        model.token_embedding.requires_grad_(False)
        model.lm_head.requires_grad_(False)

        print(
            f"are lm_head weights equal ? {torch.allclose(model.lm_head.weight, teacher_model.lm_head.weight)}"
        )
        print(
            f"are embedding weights equal ? {torch.allclose(model.token_embedding.weight, teacher_model.model.embed_tokens.weight)}"
        )

        print(f"xLSTM lm_head requires grad ? {model.lm_head.weight.requires_grad}")
        print(
            f"xLSTM embedding requires grad ? {model.token_embedding.weight.requires_grad}"
        )

        print(f"Model number of parameters: \n{count_parameters(model)}")
        print(
            f"Model number  trainable parameters: \n{count_trainable_parameters(model)}"
        )

        return model

    @staticmethod
    def from_safetensors(
        hf_repo: str,
        filename: Path | str,
        device: str = "cuda",
    ) -> "DistilxLSTM":
        """
        Creates an instance of DistilxLSTM by loading its safetensors checkpoint downloaded from the Hugging Face Hub
        and using its configuration to initialize the model.


        Parameters
        ----------
        hf_repo : str
            Hugging Face repository where the model weights are stored as well as the configuration to be used.
        filename : Path | str
            Path to the safetensors checkpoint file.
        device : str, optional
            The device on which the model will be loaded, by default "cpu"

        Returns
        -------
        DistilxLSTM

        Raises
        ------
        FileNotFoundError
            If the file does not exist on the disk.
        """
        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist on the disk.")

        config = DistilxLSTMConfig.from_pretrained(hf_repo)
        model = DistilxLSTM(config=config)
        safetensors.torch.load_model(model=model, filename=filename, device=device)

        model = model.to(device)

        return model
