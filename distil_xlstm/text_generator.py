from typing import Literal

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .modeling import DistilxLSTM

DecodingStrategy = Literal["greedy", "sampling", "top_k", "top_p", "beam_search"]


class TextGenerator:
    """
    A wrapper around :class:`DistilxLSTM` that supports various decoding strategies
    """

    def __init__(self, model: DistilxLSTM):
        self.model = model

    @torch.no_grad
    def generate(
        self,
        input_ids: torch.Tensor,
        strategy: DecodingStrategy = "greedy",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        replacement: bool = True,
        top_k: int = 10,
    ):
        self.model.eval()
        input_ids = input_ids.clone().to(self.model.device)

        match strategy:
            case "greedy":
                return self._greedy_decode(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                )
            case "sampling":
                return self._sampling_decode(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    replacement=replacement,
                )
            case "top_k":
                return self._top_k_decode(
                    input_ids=input_ids,
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

    def _greedy_decode(self, input_ids: torch.Tensor, max_new_tokens: int):
        for _ in tqdm(range(max_new_tokens)):
            logits = self.model(input_ids).logits
            next_token_probs = F.softmax(logits[:, -1, :], dim=-1)

            next_token = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def _sampling_decode(
        self,
        input_ids: torch.Tensor,
        temperature: float,
        max_new_tokens: int,
        replacement: bool = True,
    ):
        for _ in tqdm(range(max_new_tokens)):
            logits = self.model(input_ids).logits / temperature
            probs = F.softmax(logits[:, -1, :], dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1,
                replacement=replacement,
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def _top_k_decode(
        self,
        input_ids: torch.Tensor,
        top_k: int,
        max_new_tokens: int,
        temperature: float,
    ):
        for _ in tqdm(range(max_new_tokens)):
            logits = self.model(input_ids).logits
            scaled_logits = logits[:, -1, :] / temperature

            # Get top-k values and indices
            top_k_values, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)

            # Create a mask to zero out probabilities outside top-k
            top_k_mask = torch.zeros_like(scaled_logits).scatter_(1, top_k_indices, 1)

            masked_logits = scaled_logits * top_k_mask
            next_token_probs = F.softmax(masked_logits, dim=-1)
            predicted_token = torch.multinomial(next_token_probs, num_samples=1)

            input_ids = torch.cat([input_ids, predicted_token], dim=-1)

        return input_ids

    def _beam_search_decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        beam_size: int,
        temperature: float,
    ):
        batch_size = input_ids.size(0)

        for idx in tqdm(range(max_new_tokens)):
            logits = self.model(input_ids).logits
            # Scale logits by temperature
            scaled_logits = logits[:, -1, :] / temperature

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
                next_token_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

                # Update the beam
                beam[:, :, -1] = top_k_indices
                beam_probs = next_token_probs

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
                next_token_probs = F.softmax(masked_logits, dim=-1)

                # Update the beam
                beam = torch.cat([expanded_beam, top_k_indices.unsqueeze(-1)], dim=-1)
                beam_probs = torch.cat([expanded_probs, next_token_probs], dim=-1)

                # Sort the beam
                beam, beam_probs = self._sort_beam(beam, beam_probs)

                # Trim the beam
                beam = beam[:, :, :beam_size]
                beam_probs = beam_probs[:, :beam_size]

            # Check if the beam is complete
            if torch.all(beam[:, :, -1] == self.config.eos_token_id):
                break

        return beam
