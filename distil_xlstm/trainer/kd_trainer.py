from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.trainer.trainer_arguments import KDArguments


def cka_loss(teacher_h: torch.Tensor, student_h: torch.Tensor):
    """
    Computes the CKA loss between teacher and student hidden states using the HSIC approach.

    Args:
        teacher_h (torch.Tensor): Hidden states from the teacher model of shape (batch_size, hidden_dim).
        student_h (torch.Tensor): Hidden states from the student model of shape (batch_size, hidden_dim).

    Returns:
        torch.Tensor: Scalar loss representing the CKA similarity between the two tensors.
    """

    # Reshape all tensors to be 2D
    teacher_h = teacher_h.reshape(-1, teacher_h.shape[-1])
    student_h = student_h.reshape(-1, student_h.shape[-1])

    def center_matrix(x):
        """Centers a matrix by subtracting its mean."""
        mean = x.mean(dim=0, keepdim=True)
        return x - mean

    def hsic(x, y):
        """Computes the Hilbert-Schmidt Independence Criterion (HSIC) between two matrices."""
        n = x.size(0)
        assert n == y.size(0), "Input matrices must have the same number of samples"

        # Center the matrices
        x_centered = center_matrix(x)
        y_centered = center_matrix(y)

        # Compute Gram matrices
        gram_x = x_centered @ x_centered.T
        gram_y = y_centered @ y_centered.T

        # Compute HSIC
        hsic_value = (gram_x * gram_y).sum() / (n**2)
        return hsic_value

    # Compute HSIC for teacher and student hidden states
    hsic_teacher_student = hsic(teacher_h, student_h)
    hsic_teacher_teacher = hsic(teacher_h, teacher_h)
    hsic_student_student = hsic(student_h, student_h)

    # Compute CKA similarity
    cka_similarity = hsic_teacher_student / (
        torch.sqrt(hsic_teacher_teacher * hsic_student_student) + 1e-8
    )
    cka_loss = 1 - cka_similarity  # Loss is 1 - similarity

    return cka_loss


class KDTrainer(Trainer):
    def __init__(
        self,
        teacher_model: AutoModelForCausalLM,
        student_model: DistilxLSTM,
        args: KDArguments,
        tokenizer: AutoTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(
            args=args,
            model=student_model,
            processing_class=tokenizer,
            **kwargs,
        )

        self.args = args
        self.teacher = teacher_model
        self.kl_loss_fn = partial(F.kl_div, reduction="batchmean")

    @torch.no_grad()
    def _teacher_forward(self, inputs) -> CausalLMOutputWithPast:
        output = self.teacher(**inputs, output_hidden_states=True)
        return output

    def compute_loss(self, model: DistilxLSTM, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss as a combination of student cross-entropy loss, knowledge distillation loss, and CKA loss.

        Parameters
        ----------
        model : _type_
            _description_
        inputs : _type_
            _description_
        return_outputs : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """

        student_output: CausalLMOutputWithPast = model(
            **inputs,
            output_hidden_states=True,
        )
        student_logits = rearrange(student_output.logits, "b s d -> (b s) d")

        teacher_output: CausalLMOutputWithPast = self._teacher_forward(inputs)
        teacher_logits = rearrange(teacher_output.logits.detach(), "b s d -> (b s) d")

        # Compute the cross-entropy loss
        T = self.args.temperature
        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # Compute KL divergence loss
        kl_loss = self.kl_loss_fn(input=student_probs, target=teacher_probs)

        # computing the CKA loss
        avg_teacher_hidden_states = torch.cat(teacher_output.hidden_states).mean(
            dim=0,
            keepdim=True,
        )

        cka_loss_value = cka_loss(
            avg_teacher_hidden_states,
            student_output.hidden_states,
        )

        # Compute the total loss
        alpha = self.args.alpha
        beta = self.args.beta
        ce_loss = student_output.loss
        scaled_temperature = T**2

        ce_loss_term = (1 - alpha) * ce_loss
        kl_loss_term = beta * scaled_temperature * kl_loss
        cka_loss_term = alpha * cka_loss_value

        total_loss = ce_loss_term + kl_loss_term + cka_loss_term

        self.log(
            {
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "temperature": T,
                "alpha": alpha,
                "beta": beta,
                "cka_loss": cka_loss_value.item(),
            }
        )

        return (total_loss, student_output) if return_outputs else total_loss
