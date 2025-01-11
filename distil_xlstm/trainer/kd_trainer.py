from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.optim.loss import FrobeniusLoss
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
        self.frobenius_loss = FrobeniusLoss()

    @torch.no_grad()
    def _teacher_forward(self, inputs) -> CausalLMOutputWithPast:
        output = self.teacher(**inputs, output_hidden_states=True)
        return output

    def compute_loss(self, model: DistilxLSTM, inputs, return_outputs=False, **kwargs):
        """
        Compute the loss as a combination of student cross-entropy loss, knowledge distillation loss, and Frobenius loss.

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
        # student_logits = rearrange(student_output.logits, "b s d -> (b s) d")

        teacher_output: CausalLMOutputWithPast = self._teacher_forward(inputs)
        # teacher_logits = rearrange(teacher_output.logits.detach(), "b s d -> (b s) d")

        # # Compute KL divergence loss
        # T = self.args.temperature
        # student_probs = F.log_softmax(student_logits / T, dim=-1)
        # teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        # kl_loss = self.kl_loss_fn(input=student_probs, target=teacher_probs)

        # scaled_temperature = T**2
        # kl_loss_term = self.args.kl_weight * scaled_temperature * kl_loss

        # Compute Frobenius loss
        frobenius_loss: torch.Tensor = self.frobenius_loss(
            teacher_hidden_state=teacher_output.hidden_states,
            student_hidden_state=student_output.hidden_states,
        )

        frobenius_loss_term = self.args.frobenius_weight * frobenius_loss

        ce_loss = student_output.loss
        ce_loss_term = (1 - self.args.frobenius_weight) * ce_loss

        total_loss = ce_loss_term + frobenius_loss_term

        # Compute MSE loss between the hidden states of the teacher and student
        avg_teacher = rearrange(
            teacher_output.hidden_states.detach(),
            "(n b) s d -> b n s d",
            b=student_output.hidden_states.shape[0],
        ).mean(dim=1)

        mse_loss = F.mse_loss(
            student_output.hidden_states.detach(),
            avg_teacher,
        )

        self.log(
            {
                "ce_loss": ce_loss.item(),
                "frobenius_loss": frobenius_loss.item(),
                "total_loss": total_loss.item(),
                "frobenius_weight": self.args.frobenius_weight,
                "mse_loss": mse_loss.item(),
            }
        )

        return (total_loss, student_output) if return_outputs else total_loss
