from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.trainer.trainer_arguments import KDArguments


def cka_loss(teacher_reps: torch.Tensor, student_reps: torch.Tensor):
    """
    Compute the Centered Kernel Alignment (CKA) loss between teacher and student representations.

    Parameters
    ----------
    teacher_reps : torch.Tensor
        Hidden states from the teacher model (shape: [batch_size, sequence_length, hidden_size]).
    student_reps : torch.Tensor
        Hidden states from the student model (shape: [batch_size, sequence_length, hidden_size]).

    Returns
    -------
    torch.Tensor
        The CKA loss.
    """
    # Reshape the tensors to 2D: [batch_size * sequence_length, hidden_size]
    teacher_reps = teacher_reps.view(-1, teacher_reps.size(-1))  # Flatten to 2D
    student_reps = student_reps.view(-1, student_reps.size(-1))  # Flatten to 2D

    # Center the representations
    teacher_reps = teacher_reps - teacher_reps.mean(dim=0)
    student_reps = student_reps - student_reps.mean(dim=0)

    # Compute the Gram matrices
    gram_teacher = torch.matmul(teacher_reps, teacher_reps.t())
    gram_student = torch.matmul(student_reps, student_reps.t())

    # Compute the HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = torch.norm(torch.matmul(gram_teacher, gram_student), p="fro") ** 2
    norm_teacher = torch.norm(gram_teacher, p="fro") ** 2
    norm_student = torch.norm(gram_student, p="fro") ** 2

    # Compute CKA
    cka = hsic / (norm_teacher * norm_student)

    # Return the CKA loss (1 - CKA to minimize)
    return 1 - cka


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
            **inputs, output_hidden_states=True
        )

        student_logits = rearrange(student_output.logits, "b s d -> (b s) d")
        student_last_hidden_states = student_output.hidden_states[-1]

        teacher_output: CausalLMOutputWithPast = self._teacher_forward(inputs)
        teacher_logits = rearrange(teacher_output.logits.detach(), "b s d -> (b s) d")
        teacher_last_hidden_states = teacher_output.hidden_states[-1].detach()

        # Compute the cross-entropy loss
        temperature = self.args.temperature
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Compute KL divergence loss
        kl_loss = self.kl_loss_fn(input=student_probs, target=teacher_probs)

        # Compute CKA loss
        cka_loss_value = cka_loss(
            teacher_last_hidden_states, student_last_hidden_states
        )

        alpha = self.args.alpha  # Weight for CKA loss
        beta = self.args.beta  # Weight for KL loss
        ce_loss = student_output.loss
        scaled_temperature = temperature**2

        ce_loss_term = (1 - alpha) * ce_loss
        kl_loss_term = beta * scaled_temperature * kl_loss
        cka_loss_term = alpha * cka_loss_value

        total_loss = ce_loss_term + kl_loss_term + cka_loss_term

        self.log(
            {
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "cka_loss": cka_loss_value.item(),
                "total_loss": total_loss.item(),
                "temperature": temperature,
                "alpha": alpha,
                "beta": beta,
            }
        )

        return (total_loss, student_output) if return_outputs else total_loss
