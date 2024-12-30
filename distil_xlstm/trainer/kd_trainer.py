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
