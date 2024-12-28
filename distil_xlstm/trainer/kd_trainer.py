from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.trainer.trainer_arguments import KDArguments


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

    @torch.no_grad
    def _teacher_forward(self, inputs) -> CausalLMOutputWithPast:
        output = self.teacher(**inputs, output_hidden_states=True)
        return output

    def compute_loss(self, model: DistilxLSTM, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss as a combination of student cross-entropy loss and knowledge distillation loss

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

        student_output: CausalLMOutputWithPast = model(**inputs)
        student_logits = rearrange(student_output.logits, "b s d -> (b s) d")

        teacher_output: CausalLMOutputWithPast = self._teacher_forward(inputs)
        teacher_logits = rearrange(teacher_output.logits.detach(), "b s d -> (b s) d")

        # Compute the cross-entropy loss
        temperature = self.args.temperature
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # Compute KL divergence loss
        kl_loss = self.kl_loss_fn(input=student_probs, target=teacher_probs)

        alpha = self.args.alpha
        ce_loss = student_output.loss
        scaled_temperature = temperature**2

        total_loss: torch.Tensor = (
            1 - alpha
        ) * ce_loss + alpha * scaled_temperature * kl_loss

        self.log(
            {
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "temperature": temperature,
                "alpha": alpha,
            }
        )

        return (total_loss, student_output) if return_outputs else total_loss

    def _cka_loss(
        self,
        teacher_output: CausalLMOutputWithPast,
        student_output: CausalLMOutputWithPast,
    ) -> torch.Tensor:
        """
        Computes the Linear CKA loss between teacher and student hidden states.

        Parameters
        ----------
        teacher_output : CausalLMOutputWithPast
            Teacher model outputs containing hidden states.
        student_output : CausalLMOutputWithPast
            Student model outputs containing hidden states.

        Returns
        -------
        torch.Tensor
            CKA similarity loss (lower means higher alignment).
        """
        teacher_hidden_states = torch.stack(
            teacher_output.hidden_states, dim=1
        )  # [B, L, D]
        student_hidden_states = torch.stack(
            student_output.hidden_states, dim=1
        )  # [B, L, D]

        # Reshape hidden states for batch alignment
        teacher_hidden_states = rearrange(teacher_hidden_states, "b l d -> b (l d)")
        student_hidden_states = rearrange(student_hidden_states, "b l d -> b (l d)")

        # Compute Gram matrices
        gram_teacher = teacher_hidden_states @ teacher_hidden_states.T  # [B, B]
        gram_student = student_hidden_states @ student_hidden_states.T  # [B, B]

        # Center the Gram matrices
        gram_teacher = (
            gram_teacher
            - gram_teacher.mean(dim=0, keepdim=True)
            - gram_teacher.mean(dim=1, keepdim=True)
            + gram_teacher.mean()
        )
        gram_student = (
            gram_student
            - gram_student.mean(dim=0, keepdim=True)
            - gram_student.mean(dim=1, keepdim=True)
            + gram_student.mean()
        )

        # Compute CKA similarity
        cka_numerator = torch.norm(gram_teacher * gram_student, p="fro") ** 2
        cka_denominator = torch.norm(gram_teacher, p="fro") * torch.norm(
            gram_student, p="fro"
        )
        cka_similarity = cka_numerator / cka_denominator

        # CKA loss: 1 - similarity (to minimize loss)
        cka_loss = 1 - cka_similarity
        return cka_loss
