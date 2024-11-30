import copy
from pathlib import Path
from typing import Optional

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from xlstm import xLSTMBlockStack

from distil_xlstm.config import DistilxLSTMConfig
from distil_xlstm.utils import count_parameters, count_trainable_parameters


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
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        input_ids = self.token_embedding(input_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            input_ids = input_ids * attention_mask

        hidden_state = self.embedding_dropout(input_ids)
        hidden_state = self.xlstm_block_stack(input_ids)
        logits: torch.Tensor = self.lm_head(hidden_state)

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

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            hidden_states=hidden_state,
            attentions=None,
        )

    @staticmethod
    def init_for_distillation(
        *,
        teacher_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        xlstm_config_path: str,
        return_xlstm_config: bool = False,
    ):
        # freezing the teacher
        for param in teacher_model.parameters():
            param.requires_grad_(False)

        teacher_config = teacher_model.config
        with open(xlstm_config_path, "r") as file:
            xlstm_config_dict = yaml.safe_load(file)
            xlstm_config_dict["vocab_size"] = teacher_config.vocab_size
            xlstm_config_dict["embedding_dim"] = teacher_config.hidden_size

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
    ) -> "DistilxLSTM":
        model, config = DistilxLSTM.init_for_distillation(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            xlstm_config_path=xlstm_config_path,
            return_xlstm_config=True,
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

    @torch.no_grad()
    def generate(
        self,
        encoding,
        max_new_tokens: int = 50,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ):
        """
        Generate text using different sampling strategies.

        Parameters:
        -----------
        encoding : dict
            Tokenizer encoding of the input prompt
        max_new_tokens : int
            Maximum number of tokens to generate
        strategy : str, optional
            Generation strategy. Options:
            - 'greedy': Always choose the most probable token
            - 'top_k': Sample from top k most probable tokens
            - 'top_p': Sample from tokens with cumulative probability <= top_p
        temperature : float, optional
            Temperature for softmax scaling. Lower values make sampling more deterministic
        top_k : int, optional
            Number of top tokens to consider for top-k sampling
        top_p : float, optional
            Cumulative probability threshold for top-p (nucleus) sampling

        Returns:
        --------
        list
            Generated token ids
        """
        predictions = []
        input_ids = encoding["input_ids"]

        def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
            """Filter logits based on top-k and top-p strategies."""
            # Top-k filtering
            if top_k > 0:
                # Remove tokens with low probability
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
                logits = torch.where(logits < min_values, float("-inf"), logits)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                # Sort and compute cumulative probabilities
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter the indices back to their original positions
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float("-inf")

            return logits

        for _ in range(max_new_tokens):
            # Get logits for the current input
            logits = self(input_ids=input_ids, **encoding).logits

            # Extract logits for the last token
            last_token_logits = logits[:, -1, :]

            # Apply temperature scaling
            last_token_logits = last_token_logits / temperature

            # Apply sampling strategy
            if strategy == "greedy":
                # Always choose the most probable token
                predicted_token = torch.argmax(last_token_logits, dim=-1)
            else:
                # Apply top-k and top-p filtering
                filtered_logits = top_k_top_p_filtering(
                    last_token_logits.unsqueeze(0), top_k=top_k, top_p=top_p
                ).squeeze(0)

                # Sample from the filtered distribution
                probs = F.softmax(filtered_logits, dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1)

            # Store and append the predicted token
            predictions.append(predicted_token.item())
            input_ids = torch.cat([input_ids, predicted_token.unsqueeze(0)], dim=-1)

        return predictions
