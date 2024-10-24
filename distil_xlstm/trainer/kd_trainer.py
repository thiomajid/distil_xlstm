import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import AutoModelForCausalLM, Trainer
from transformers.modeling_outputs import CausalLMOutput

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.trainer.trainer_arguments import KDArguments


class KDTrainer(Trainer):
    def __init__(
        self,
        teacher_model: AutoModelForCausalLM,
        student_model: DistilxLSTM,
        args: KDArguments,
        **kwargs,
    ) -> None:
        super().__init__(
            args=args,
            model=student_model,
            **kwargs,
        )

        self.args = args
        self.teacher = teacher_model
        self.teacher.eval()
        self.student = student_model

        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    @torch.no_grad
    def _teacher_forward(self, inputs) -> CausalLMOutput:
        output = self.teacher(**inputs)
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

        student_output: CausalLMOutput = self.student(inputs["input_ids"])

        student_logits = rearrange(student_output.logits, "b s d -> (b s) d")

        teacher_logits = self._teacher_forward(inputs).logits.detach()
        teacher_logits = rearrange(teacher_logits, "b s d -> (b s) d")

        # Reshape labels to match the logits shape
        labels: torch.Tensor = inputs["labels"]
        labels = rearrange(labels, "b s -> (b s)")

        # Compute the cross-entropy loss
        ce_loss = self.ce_loss_fn(student_logits, labels)
        student_probs = F.log_softmax(student_logits / self.args.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.args.temperature, dim=-1)

        # Compute KL divergence loss
        kd_loss = self.kl_loss_fn(student_probs, teacher_probs) * (
            self.args.temperature**2
        )

        total_loss: torch.Tensor = (
            self.args.ce_weight * ce_loss + self.args.kl_weight * kd_loss
        )

        return (total_loss, student_output) if return_outputs else total_loss
