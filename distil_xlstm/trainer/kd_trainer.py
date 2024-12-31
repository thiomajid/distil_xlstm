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

        # Compute KL divergence loss
        T = self.args.temperature
        student_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = self.kl_loss_fn(input=student_probs, target=teacher_probs)

        # Compute Frobenius loss
        frobenius_loss = self._frobenius_loss(
            teacher_hidden_state=teacher_output.hidden_states,
            student_hidden_state=student_output.hidden_states,
        )

        alpha = self.args.alpha
        fro_weight = self.args.frobenius_weight
        ce_loss = student_output.loss
        scaled_temperature = T**2

        ce_loss_term = (1 - fro_weight) * ce_loss
        kl_loss_term = alpha * scaled_temperature * kl_loss
        frobenius_loss_term = fro_weight * frobenius_loss

        total_loss = ce_loss_term + kl_loss_term + frobenius_loss_term

        self.log(
            {
                "ce_loss": ce_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "temperature": T,
                "alpha": alpha,
                "frobenius_weight": fro_weight,
                "frobenius_loss": frobenius_loss.item(),
            }
        )

        return (total_loss, student_output) if return_outputs else total_loss

    def _frobenius_loss(
        self,
        teacher_hidden_state: torch.Tensor,
        student_hidden_state: torch.Tensor,
    ):
        if isinstance(teacher_hidden_state, tuple):
            teacher_hidden_state = torch.cat(teacher_hidden_state, dim=0)

        avg_teacher_hidden_state = teacher_hidden_state.mean(dim=0, keepdim=True)

        # Repeat the averaged teacher hidden state to match the student's batch size
        avg_teacher_hidden_state = avg_teacher_hidden_state.repeat(
            student_hidden_state.shape[0], 1, 1
        )

        if student_hidden_state.shape != avg_teacher_hidden_state.shape:
            raise ValueError(
                f"Shape mismatch: student hidden state has shape {student_hidden_state.shape}, "
                f"but averaged teacher hidden state has shape {avg_teacher_hidden_state.shape}."
            )

        norm = torch.norm(avg_teacher_hidden_state - student_hidden_state, p="fro") ** 2

        return norm
