import math
from typing import Literal

import torch
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
        teacher_hidden_state: torch.Tensor,
        student_hidden_state: torch.Tensor,
        computation: FrobeniusNormComputation = "ratio",
    ):
        if isinstance(teacher_hidden_state, tuple):
            teacher_hidden_state = torch.cat(teacher_hidden_state, dim=0)

        if computation == "ratio":
            n_layers, batch_size, sequence_length, d_model = student_hidden_state.shape
            n_teacher_layers = teacher_hidden_state.shape[0]

            step_size = n_teacher_layers // n_layers

            frobenius_norms = torch.empty_like(
                student_hidden_state, device=student_hidden_state.device
            )

            for idx in range(n_layers):
                target_representation = teacher_hidden_state[
                    idx * step_size : (idx + 1) * step_size
                ]

                target_representation = target_representation.mean(dim=0)

                if student_hidden_state.shape != target_representation.shape:
                    raise ValueError(
                        f"Shape mismatch: student hidden state has shape {student_hidden_state.shape}, "
                        f"but averaged teacher hidden state has shape {target_representation.shape}."
                    )

                frobenius_norm = torch.norm(
                    target_representation - student_hidden_state[idx],
                    p="fro",
                )
                frobenius_norms[idx] = frobenius_norm

            norm = frobenius_norms.mean(dim=0)
        else:
            # layer-wise average of teacher hidden states
            # (num_layers, batch_size, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
            avg_teacher_hidden_state = teacher_hidden_state.mean(dim=0)

            if student_hidden_state.shape != avg_teacher_hidden_state.shape:
                raise ValueError(
                    f"Shape mismatch: student hidden state has shape {student_hidden_state.shape}, "
                    f"but averaged teacher hidden state has shape {avg_teacher_hidden_state.shape}."
                )

            norm = torch.norm(avg_teacher_hidden_state - student_hidden_state, p="fro")

        # normalize by \sqrt{num_elements} prevents the loss from being too
        # large or too small especially for large tensors
        norm = norm / math.sqrt(student_hidden_state.numel())

        return norm
