import math
from typing import Literal

import torch
from einops import rearrange
from torch import nn

FrobeniusNormComputation = Literal["average", "ratio"]


class FrobeniusLoss(nn.Module):
    """

    FrobeniusLoss is a custom loss function that computes the Frobenius norm
    between the hidden states of a teacher model and a student model. This loss
    function is typically used in knowledge distillation tasks where the goal
    is to train a student model to mimic the behavior of a teacher model.

    :math: `L_{Frob} = \frac{1}{\sqrt{N}} \left\| \frac{1}{L} \sum_{l=1}^{L} h_{T}^{(l)} - h_{S}^{(l)} \right\|_{F}`

    Methods
    -------
        forward(teacher_hidden_state: torch.Tensor, student_hidden_state: torch.Tensor) -> torch.Tensor
            Computes the Frobenius norm between the teacher's and student's hidden states.
    Parameters
    ----------
        teacher_hidden_state : torch.Tensor
            The hidden states from the teacher model. Can be a tuple of tensors.
        student_hidden_state : torch.Tensor
            The hidden states from the student model.
    Returns
    -------
        torch.Tensor
            The computed Frobenius norm, normalized by the number of elements in the tensor.
    Raises
    ------
        ValueError
            If the shapes of the student hidden state and the averaged teacher hidden state do not match.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        teacher_hidden_states: torch.Tensor,
        student_hidden_states: torch.Tensor,
        computation: FrobeniusNormComputation = "ratio",
    ):
        if isinstance(teacher_hidden_states, tuple):
            teacher_hidden_states = torch.cat(teacher_hidden_states, dim=0)

        batch_size = student_hidden_states.shape[0]
        teacher_hidden_states = rearrange(
            teacher_hidden_states,
            "(n b) s d -> b n s d",
            b=batch_size,
        )

        norm_per_block: torch.Tensor | None = None
        if computation == "ratio":
            num_xlstm_blocks = student_hidden_states.shape[0]
            num_attention_layers = teacher_hidden_states.shape[1]

            # Calculate base step size
            step_size = num_attention_layers // num_xlstm_blocks

            norm_per_block = torch.empty(
                num_xlstm_blocks,
                device=student_hidden_states.device,
                dtype=student_hidden_states.dtype,
                requires_grad=True,
            )

            for idx in range(num_xlstm_blocks):
                student_representation = student_hidden_states[idx]

                # Standard case for most blocks
                start_idx = idx * step_size

                # For the last block, include all remaining layers
                if idx == num_xlstm_blocks - 1:
                    end_idx = num_attention_layers
                else:
                    end_idx = (idx + 1) * step_size

                # Select teacher layers for current block
                target_representation = teacher_hidden_states[:, start_idx:end_idx]

                # Average over the step_size dimension
                target_representation = target_representation.mean(dim=1)

                if student_representation.shape != target_representation.shape:
                    raise ValueError(
                        f"Shape mismatch: student hidden state has shape {student_hidden_states[idx].shape}, "
                        f"but averaged teacher hidden state has shape {target_representation.shape}."
                    )

                block_norm = torch.norm(
                    target_representation - student_representation,
                    p="fro",
                )

                norm_per_block[idx] = block_norm

            norm = norm_per_block.mean(dim=0)
        else:
            # layer-wise average of teacher hidden states
            # (batch_size, num_layers, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
            avg_teacher_hidden_state = teacher_hidden_states.mean(dim=1)

            if student_hidden_states.shape != avg_teacher_hidden_state.shape:
                raise ValueError(
                    f"Shape mismatch: student hidden state has shape {student_hidden_states.shape}, "
                    f"but averaged teacher hidden state has shape {avg_teacher_hidden_state.shape}."
                )

            norm = torch.norm(avg_teacher_hidden_state - student_hidden_states, p="fro")

        # normalize by \sqrt{num_elements} prevents the loss from being too
        # large or too small especially for large tensors
        norm = norm / math.sqrt(student_hidden_states.numel())

        return norm, norm_per_block
