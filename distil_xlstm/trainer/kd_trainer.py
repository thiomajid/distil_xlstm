import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import AutoModelForCausalLM, Trainer
from xlstm import xLSTMLMModel

from distil_xlstm.config import KDArguments


class KDTrainer(Trainer):
    def __init__(
        self,
        teacher_model: AutoModelForCausalLM,
        student_model: xLSTMLMModel,
        kd_config: KDArguments,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model=student_model, *args, **kwargs)

        self.config = kd_config

        self.teacher = teacher_model
        self.teacher.eval()
        self.student = student_model

        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    @torch.no_grad
    def _teacher_forward(self, inputs) -> torch.Tensor:
        output = self.teacher(**inputs)
        return output.logits

    def compute_loss(
        self,
        model: xLSTMLMModel,
        inputs,
        return_outputs=False,
    ):
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

        student_logits = self.student(inputs["input_ids"])
        student_logits = rearrange(student_logits, "b s d -> (b s) d")

        teacher_logits = self._teacher_forward(inputs).detach()
        teacher_logits = rearrange(teacher_logits, "b s d -> (b s) d")

        # Reshape labels to match the logits shape
        labels: torch.Tensor = inputs["labels"]
        labels = rearrange(labels, "b s -> (b s)")

        # Compute the cross-entropy loss
        ce_loss = self.ce_loss_fn(student_logits, labels)

        student_probs = F.log_softmax(student_logits / self.config.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.config.temperature, dim=-1)

        # Compute KL divergence loss
        kd_loss = self.kl_loss_fn(student_probs, teacher_probs) * (
            self.config.temperature**2
        )

        total_loss: torch.Tensor = (
            self.config.ce_weight * ce_loss + self.config.kl_weight * kd_loss
        )

        return total_loss
