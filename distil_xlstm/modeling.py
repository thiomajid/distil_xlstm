from pathlib import Path
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from xlstm import xLSTMBlockStack
from xlstm.components.init import small_init_init_

from distil_xlstm.config import DistilxLSTMConfig
from distil_xlstm.utils import (
    DistilxLSTMCausalLMOutput,
    count_parameters,
    count_trainable_parameters,
    next_multiple_of,
)


class DistilxLSTMModel(PreTrainedModel):
    def __init__(self, config: DistilxLSTMConfig) -> None:
        super().__init__(config)

        self.config = config

        # Code from original xLSTMLMModel __init__ method
        self.embedding = nn.Embedding(
            num_embeddings=config.xlstm_config.vocab_size,
            embedding_dim=config.xlstm_config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        self.embedding_dropout = (
            nn.Dropout(config.xlstm_config.dropout)
            if config.xlstm_config.add_embedding_dropout
            else nn.Identity()
        )

        self.backbone = xLSTMBlockStack(config=config.xlstm_config)

    def forward(self, x: torch.Tensor):
        h_t = self.embedding(x)
        h_t = self.embedding_dropout(h_t)

        for block in self.backbone:
            h_t = block(h_t)

        h_t = self.backbone.post_blocks_norm(h_t)

        return h_t

    def reset_parameters(self):
        small_init_init_(
            self.embedding.weight,
            dim=self.config.xlstm_config.embedding_dim,
        )

        if not isinstance(self.backbone.post_blocks_norm, nn.Identity):
            self.backbone.post_blocks_norm.reset_parameters()

        for block in self.backbone.blocks:
            block.reset_parameters()

    def reset_parameters_for_distillation(self):
        for block in self.backbone.blocks:
            block.reset_parameters()

        if not isinstance(self.backbone.post_blocks_norm, nn.Identity):
            self.backbone.post_blocks_norm.reset_parameters()


class DistilxLSTMForCausalLM(PreTrainedModel):
    config_class = DistilxLSTMConfig

    def __init__(self, config: DistilxLSTMConfig) -> None:
        super().__init__(config)

        self.config = config
        self.num_blocks = config.xlstm_config.num_blocks

        self.xlstm = DistilxLSTMModel(config=config)
        self.lm_head = nn.Linear(
            in_features=config.xlstm_config.embedding_dim,
            out_features=config.xlstm_config.vocab_size,
            bias=False,
        )

        if config.xlstm_config.tie_weights:
            self.lm_head.weight = self.xlstm.embedding.weight

    def reset_parameters(self):
        self.xlstm.reset_parameters()

        if not self.config.xlstm_config.tie_weights:
            small_init_init_(
                self.lm_head.weight, dim=self.config.xlstm_config.embedding_dim
            )

    def reset_parameters_for_distillation(self):
        self.xlstm.reset_parameters_for_distillation()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> DistilxLSTMCausalLMOutput:
        h_t: torch.Tensor = self.xlstm.embedding(input_ids)
        h_t = self.xlstm.embedding_dropout(h_t)

        hidden_states = () if output_hidden_states else None

        for block in self.xlstm.backbone.blocks:
            block_state = block(h_t)
            h_t = block_state
            if output_hidden_states:
                hidden_states = hidden_states + (block_state,)

        h_t = self.xlstm.backbone.post_blocks_norm(h_t)
        logits = self.lm_head(h_t)

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
            )

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": h_t,
            "hidden_states": hidden_states,
        }

    @staticmethod
    def init_for_distillation(
        *,
        config: DistilxLSTMConfig,
        teacher_config: AutoConfig,
        tokenizer: AutoTokenizer,
    ):
        config.xlstm_config.vocab_size = teacher_config.vocab_size
        config.xlstm_config.embedding_dim = teacher_config.hidden_size

        if config.num_blocks_init == "same":
            config.xlstm_config.num_blocks = teacher_config.num_hidden_layers
        elif config.num_blocks_init == "half":
            config.xlstm_config.num_blocks = teacher_config.num_hidden_layers // 2
        elif config.num_blocks_init == "custom":
            pass
        else:
            raise ValueError(
                f"num_blocks_init should be one of ['same', 'half', 'custom'], but got {config.num_blocks_init}"
            )

        rounded_teacher_num_heads = next_multiple_of(
            teacher_config.num_attention_heads, multiple=4
        )

        # Having too many heads is not a problem per se, but it makes the hidden dimensions
        # too small but also increases the number of computational units
        # dividing by 4 is a good trade-off
        num_heads = rounded_teacher_num_heads // 4
        config.xlstm_config.mlstm_block.mlstm.num_heads = num_heads
        config.xlstm_config.slstm_block.slstm.num_heads = num_heads

        config.pad_token_id = tokenizer.pad_token_id
        model = DistilxLSTMForCausalLM(config=config)
        model.reset_parameters_for_distillation()

        return model

    @staticmethod
    def init_for_distillation_with_freezed_head_and_embedding(
        *,
        config: DistilxLSTMConfig,
        teacher_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ) -> "DistilxLSTMForCausalLM":
        model = DistilxLSTMForCausalLM.init_for_distillation(
            config=config,
            teacher_config=teacher_model.config,
            tokenizer=tokenizer,
            return_xlstm_config=True,
        )

        model = model.to(teacher_model.device)
        model.xlstm.embedding.load_state_dict(
            teacher_model.model.embed_tokens.state_dict()
        )
        model.lm_head.load_state_dict(teacher_model.lm_head.state_dict())

        if config.xlstm_config.tie_weights:
            model.lm_head.weight = model.token_embedding.weight

        model.xlstm.embedding.requires_grad_(False)
        model.lm_head.requires_grad_(False)

        print(
            f"are lm_head weights equal ? {torch.allclose(model.lm_head.weight, teacher_model.lm_head.weight)}"
        )
        print(
            f"are embedding weights equal ? {torch.allclose(model.xlstm.embedding.weight, teacher_model.model.embed_tokens.weight)}"
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
    ) -> "DistilxLSTMForCausalLM":
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
        model = DistilxLSTMForCausalLM(config=config)
        safetensors.torch.load_model(model=model, filename=filename, device=device)

        model = model.to(device)

        return model
