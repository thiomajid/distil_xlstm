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
from xlstm.components.init import small_init_init_
from xlstm.xlstm_large.model import xLSTMLargeConfig

from distil_xlstm.config import DistilxLSTMConfig
from distil_xlstm.optim.loss import FrobeniusNormComputation
from distil_xlstm.utils import (
    count_parameters,
    count_trainable_parameters,
    next_multiple_of,
    xLSTMCausalLMOutput,
)


class DistilxLSTMModel(PreTrainedModel):
    def __init__(self, config: DistilxLSTMConfig) -> None:
        super().__init__(config)

        self.config = config

        # Code from original xLSTMLMModel __init__ method
        self.embedding = nn.Embedding(
            num_embeddings=config.xlstm_cfg.vocab_size,
            embedding_dim=config.xlstm_cfg.embedding_dim,
        )

        self.embedding_dropout = (
            nn.Dropout(config.xlstm_cfg.dropout)
            if config.xlstm_cfg.add_embedding_dropout
            else nn.Identity()
        )

        self.backbone = xLSTMBlockStack(config=config.xlstm_cfg)

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
            dim=self.config.xlstm_cfg.embedding_dim,
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
        self.num_blocks = config.xlstm_cfg.num_blocks

        self.xlstm = DistilxLSTMModel(config=config)
        self.lm_head = nn.Linear(
            in_features=config.xlstm_cfg.embedding_dim,
            out_features=config.xlstm_cfg.vocab_size,
            bias=False,
        )

        if config.xlstm_cfg.tie_weights:
            self.lm_head.weight = self.xlstm.embedding.weight

    def reset_parameters(self):
        self.xlstm.reset_parameters()

        if not self.config.xlstm_cfg.tie_weights:
            small_init_init_(
                self.lm_head.weight, dim=self.config.xlstm_cfg.embedding_dim
            )

    def reset_parameters_for_distillation(self):
        self.xlstm.reset_parameters_for_distillation()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frobenius_computation: Optional[FrobeniusNormComputation] = None,
        **kwargs,
    ) -> xLSTMCausalLMOutput:
        h_t: torch.Tensor = self.xlstm.embedding(input_ids)
        h_t = self.xlstm.embedding_dropout(h_t)

        hidden_states_per_block = (
            torch.empty(
                size=(
                    self.num_blocks,
                    *h_t.shape,
                ),
                device=h_t.device,
                dtype=h_t.dtype,
            )
            if frobenius_computation == "ratio"
            else None
        )

        logits: Optional[torch.TensorBase] = None
        if frobenius_computation == "ratio":
            hidden_states_per_block_list = []
            for _, block in enumerate(self.xlstm.backbone.blocks):
                block_state = block(h_t)
                h_t = block_state  # Update hidden_states for next iteration
                hidden_states_per_block_list.append(block_state)

            # Stack the collected outputs
            hidden_states_per_block = torch.stack(hidden_states_per_block_list)
            hidden_states_per_block.requires_grad_(True)

            last_hidden_state = self.xlstm.backbone.post_blocks_norm(
                hidden_states_per_block[-1]
            )

            h_t = last_hidden_state
            logits = self.lm_head(h_t)

        elif frobenius_computation == "average":
            for block in self.xlstm.backbone.blocks:
                h_t = block(h_t)

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
            "logits": logits,
            "loss": loss,
            "hidden_states": h_t,
            "hidden_states_per_block": hidden_states_per_block,
        }

    @staticmethod
    def init_for_distillation(
        *,
        teacher_config: AutoConfig,
        tokenizer: AutoTokenizer,
        xlstm_config_path: str,
        max_sequence_length: int,
        return_xlstm_config: bool = False,
        slstm_pos: list[int] | None = None,
        v2: bool = True,
        use_tfla: bool = False,
    ):
        with open(xlstm_config_path, "r") as file:
            cfg_dict: dict = yaml.safe_load(file)

        xlstm_config_dict = cfg_dict.pop("xlstm_config")
        xlstm_config_dict["vocab_size"] = teacher_config.vocab_size
        xlstm_config_dict["embedding_dim"] = teacher_config.hidden_size
        xlstm_config_dict["context_length"] = max_sequence_length

        if v2:
            num_blocks = teacher_config.num_hidden_layers // 2
            xlstm_config_dict["num_blocks"] = num_blocks
            xlstm_config_dict["slstm_at"] = (
                slstm_pos
                if slstm_pos is not None
                else list(range(0, num_blocks - 1, 2))
            )

            teacher_num_heads = next_multiple_of(
                teacher_config.num_attention_heads, multiple=4
            )

            # Having too many heads is not a problem per se, but it makes the hidden dimensions
            # too small but also increases the number of computational units
            # dividing by 4 is a good trade-off
            num_heads = teacher_num_heads // 4

            xlstm_config_dict["mlstm_block"]["mlstm"]["num_heads"] = num_heads
            xlstm_config_dict["slstm_block"]["slstm"]["num_heads"] = num_heads

        xlstm_config = DistilxLSTMConfig.parse_xlstm_config_dict(
            copy.deepcopy(xlstm_config_dict)
        )

        tfla_config = None
        if use_tfla:
            tfla_dict = cfg_dict.get("tfla_config", {})
            tfla_config = xLSTMLargeConfig(**tfla_dict)
            tfla_config.num_blocks = xlstm_config.num_blocks
            tfla_config.embedding_dim = xlstm_config.embedding_dim
            tfla_config.vocab_size = xlstm_config.vocab_size
            tfla_config.num_heads = xlstm_config.mlstm_block.mlstm.num_heads // 4

        distil_xlstm_config = DistilxLSTMConfig(
            xlstm_cfg=xlstm_config,
            tfla_config=tfla_config,
            **cfg_dict,
        )

        distil_xlstm_config.use_tfla = use_tfla

        xlstm_config.pad_token_id = tokenizer.pad_token_id

        model = DistilxLSTMForCausalLM(config=distil_xlstm_config)
        model.reset_parameters_for_distillation()

        if return_xlstm_config:
            return model, xlstm_config

        return model

    @staticmethod
    def init_for_distillation_with_freezed_head_and_embedding(
        *,
        teacher_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        xlstm_config_path: str,
        max_sequence_length: int,
        slstm_pos: list[int] | None = None,
        v2: bool = True,
        use_tfla: bool = False,
    ) -> "DistilxLSTMForCausalLM":
        if use_tfla and not v2:
            raise ValueError(
                "TFLA is only supported for v2 initialization heuristic. Please set v2=True."
            )

        model, config = DistilxLSTMForCausalLM.init_for_distillation(
            teacher_config=teacher_model.config,
            tokenizer=tokenizer,
            xlstm_config_path=xlstm_config_path,
            return_xlstm_config=True,
            slstm_pos=slstm_pos,
            max_sequence_length=max_sequence_length,
            v2=v2,
            use_tfla=use_tfla,
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
