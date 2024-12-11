import copy
from pathlib import Path
from typing import Literal, Optional

import safetensors
import safetensors.torch
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from xlstm import xLSTMBlockStack

from distil_xlstm.config import DistilxLSTMConfig
from distil_xlstm.utils import count_parameters, count_trainable_parameters

DecodingStrategy = Literal["greedy", "sampling", "top_k", "top_p", "beam_search"]


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

    @torch.no_grad
    def generate(
        self,
        encodings,
        strategy: DecodingStrategy = "greedy",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        replacement: bool = False,
        top_k: int = 10,
    ):
        self.eval()
        encodings = encodings.to(self.device)

        match strategy:
            case "greedy":
                return self._greedy_decode(
                    encodings=encodings,
                    max_new_tokens=max_new_tokens,
                )
            case "sampling":
                return self._sampling_decode(
                    encodings=encodings,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    replacement=replacement,
                )
            case "top_k":
                return self._top_k_decode(
                    encodings=encodings,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )

            case "beam_search":
                raise NotImplementedError("Beam search decoding is not yet implemented")

            case "top_p":
                raise NotImplementedError("Top-p decoding is not yet implemented")

            case _:
                raise ValueError(f"Invalid decoding strategy: {strategy}")

    def _greedy_decode(self, encodings, max_new_tokens: int):
        input_ids: torch.Tensor = encodings["input_ids"].clone()

        for _ in tqdm(range(max_new_tokens)):
            logits = self(input_ids=input_ids, **encodings).logits
            next_token_probs = F.softmax(logits[:, -1, :], dim=-1)

            # [batch size, 1, dim] => [batch size, 1]
            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def _sampling_decode(
        self,
        encodings,
        temperature: float,
        max_new_tokens: int,
        replacement: bool = False,
    ):
        input_ids: torch.Tensor = encodings["input_ids"].clone()

        for _ in tqdm(range(max_new_tokens)):
            scaled_logits = self(input_ids=input_ids, **encodings).logits / temperature
            probs = F.softmax(scaled_logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(
                probs,
                num_samples=1,
                replacement=replacement,
            )

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def _top_k_decode(
        self,
        encodings,
        top_k: int,
        max_new_tokens: int,
        temperature: float,
    ):
        input_ids: torch.Tensor = encodings["input_ids"].clone()

        for idx in tqdm(range(max_new_tokens)):
            student_logits = self(input_ids=input_ids, **encodings).logits
            # Scale logits by temperature
            scaled_logits = student_logits[:, -1, :] / temperature

            # Get top-k values and indices
            top_k_values, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)

            # Create a mask to zero out probabilities outside top-k
            top_k_mask = torch.zeros_like(scaled_logits).scatter_(1, top_k_indices, 1)

            masked_logits = scaled_logits * top_k_mask
            student_probs = torch.nn.functional.softmax(masked_logits, dim=-1)
            predicted_token = torch.multinomial(student_probs, num_samples=1)

            input_ids = torch.cat([input_ids, predicted_token], dim=-1)

        return input_ids

    def _beam_search_decode(
        self,
        encodings,
        max_new_tokens: int,
        beam_size: int,
        temperature: float,
    ):
        input_ids: torch.Tensor = encodings["input_ids"].clone()
        batch_size = input_ids.size(0)

        for idx in tqdm(range(max_new_tokens)):
            student_logits = self(input_ids=input_ids, **encodings).logits
            # Scale logits by temperature
            scaled_logits = student_logits[:, -1, :] / temperature

            if idx == 0:
                # Initialize the beam search
                beam = torch.zeros((batch_size, beam_size, 1), dtype=torch.long).to(
                    input_ids.device
                )
                beam_probs = torch.zeros((batch_size, beam_size)).to(input_ids.device)

                # Get top-k values and indices
                top_k_values, top_k_indices = torch.topk(
                    scaled_logits, beam_size, dim=-1
                )

                # Create a mask to zero out probabilities outside top-k
                top_k_mask = torch.zeros_like(scaled_logits).scatter_(
                    1, top_k_indices, 1
                )

                masked_logits = scaled_logits * top_k_mask
                student_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

                # Update the beam
                beam[:, :, -1] = top_k_indices
                beam_probs = student_probs

            else:
                # Expand the beam
                expanded_beam = beam.unsqueeze(-1).expand(-1, -1, -1, beam_size)
                expanded_probs = beam_probs.unsqueeze(-1).expand(-1, -1, beam_size)

                # Get top-k values and indices
                top_k_values, top_k_indices = torch.topk(
                    scaled_logits, beam_size, dim=-1
                )

                # Create a mask to zero out probabilities outside top-k
                top_k_mask = torch.zeros_like(scaled_logits).scatter_(
                    1, top_k_indices, 1
                )

                masked_logits = scaled_logits * top_k_mask
                student_probs = F.softmax(masked_logits, dim=-1)

                # Update the beam
                beam = torch.cat([expanded_beam, top_k_indices.unsqueeze(-1)], dim=-1)
                beam_probs = torch.cat([expanded_probs, student_probs], dim=-1)

                # Sort the beam
                beam, beam_probs = self._sort_beam(beam, beam_probs)

                # Trim the beam
                beam = beam[:, :, :beam_size]
                beam_probs = beam_probs[:, :beam_size]

            # Check if the beam is complete
            if torch.all(beam[:, :, -1] == self.config.eos_token_id):
                break

        return beam
