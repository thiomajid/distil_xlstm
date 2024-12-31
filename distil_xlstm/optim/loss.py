import math

import torch
from einops import rearrange
from torch import nn


class FrobeniusLoss(nn.Module):
    """

    FrobeniusLoss is a custom loss function that computes the Frobenius norm
    between the hidden states of a teacher model and a student model. This loss
    function is typically used in knowledge distillation tasks where the goal
    is to train a student model to mimic the behavior of a teacher model.

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
    ):
        if isinstance(teacher_hidden_state, tuple):
            teacher_hidden_state = torch.cat(teacher_hidden_state, dim=0)

        batch_size = student_hidden_state.shape[0]

        teacher_hidden_state = rearrange(
            teacher_hidden_state,
            "(n b) s d -> b n s d",
            b=batch_size,
        )

        # layer-wise average of teacher hidden states
        # (batch_size, num_layers, seq_len, hidden_size) -> (batch_size, seq_len, hidden_size)
        avg_teacher_hidden_state = teacher_hidden_state.mean(dim=1)

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
