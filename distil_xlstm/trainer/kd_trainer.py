from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from distil_xlstm.modeling import DistilxLSTMForCausalLM
from distil_xlstm.optim.loss import FrobeniusLoss
from distil_xlstm.trainer.arguments import KDArguments
from distil_xlstm.utils import DistilxLSTMCausalLMOutput


class KDTrainer(Trainer):
    def __init__(
        self,
        teacher_model: AutoModelForCausalLM,
        student_model: DistilxLSTMForCausalLM,
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
        self.alignment_criterion = (
            self._get_alignment_criterion() if args.compute_alignment_loss else None
        )

        self.tb_writer = SummaryWriter(
            log_dir=args.logging_dir, filename_suffix="manual_logs"
        )

    def _get_alignment_criterion(self):
        if self.args.alignment_loss == "frobenius":
            return FrobeniusLoss(reduction=self.args.frobenius_norm_reduction)
        elif self.args.alignment_loss == "cosine":
            return torch.nn.CosineEmbeddingLoss(reduction="mean")
        else:
            raise ValueError(
                f"Unknown alignment loss: {self.args.alignment_loss}. "
                f"Supported losses are: frobenius"
            )

    @torch.no_grad()
    def _teacher_forward(self, inputs) -> CausalLMOutputWithPast:
        output = self.teacher(**inputs, output_hidden_states=True)
        return output

    def compute_loss(
        self,
        model: DistilxLSTMForCausalLM,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        student_output: DistilxLSTMCausalLMOutput = model(
            **inputs,
            output_hidden_states=True,
        )

        teacher_output = self._teacher_forward(inputs)

        metrics = dict()
        task_loss = student_output["loss"].mean()
        task_weight = 1
        total_loss = torch.tensor(
            0.0,
            device=task_loss.device,
            dtype=task_loss.dtype,
        )

        # # Compute KL divergence loss
        if self.args.compute_kl_loss:
            student_logits = rearrange(student_output["logits"], "b s d -> (b s) d")
            teacher_logits = rearrange(
                teacher_output.logits.detach(), "b s d -> (b s) d"
            )

            T = self.args.temperature
            student_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)

            kl_loss = self.kl_loss_fn(input=student_probs, target=teacher_probs).mean()
            total_loss += kl_loss * self.args.kl_weight * (T**2)
            task_weight -= self.args.kl_weight

            metrics.update(
                {
                    "kl_loss": kl_loss.item(),
                    "kl_weight": self.args.kl_weight,
                }
            )

        # # Compute Frobenius loss
        if self.args.compute_alignment_loss:
            student_h = student_output["hidden_states"]  # a tuple of tensors
            student_h = torch.cat(student_h, dim=0)

            teacher_h = None
            if isinstance(teacher_output.hidden_states, tuple):
                # skip the first hidden state because the "transformers" library
                # sets the embedding layer's output as the first element in the
                # all_hidden_states tuple
                teacher_h = torch.cat(teacher_output.hidden_states[1:], dim=0)
            else:
                teacher_h = teacher_output.hidden_states[1:]

            # Compute the alignment loss
            alignment_loss = None
            if self.args.alignment_loss == "frobenius":
                alignment_loss = self.alignment_criterion(
                    teacher_hidden_states=teacher_h,
                    student_hidden_states=student_h,
                )
            elif self.args.alignment_loss == "cosine":
                # Flatten the tensors from [batch, seq, hidden] to [batch*seq, hidden]
                student_h_flat = rearrange(student_h, "b s h -> (b s) h")
                teacher_h_flat = rearrange(teacher_h, "b s h -> (b s) h")

                # Create target tensor with same batch size as flattened tensors
                target = torch.ones(student_h_flat.shape[0], device=student_h.device)

                # Compute the cosine similarity between the teacher and student hidden states
                alignment_loss = self.alignment_criterion(
                    student_h_flat, teacher_h_flat, target=target
                )

            total_loss += alignment_loss.mean() * self.args.alignment_weight

            if self.args.additive_alignment_weight:
                task_weight -= self.args.alignment_weight

            metrics.update(
                {
                    "alignment_loss": alignment_loss.item(),
                    "alignment_weight": self.args.alignment_weight,
                }
            )

        total_loss += task_weight * task_loss
        perplexity = torch.exp(task_loss)
        metrics.update(
            {
                "ce_weight": task_weight,
                "ce_loss": task_loss.item(),
                "perplexity": perplexity.item(),
                "total_loss": total_loss.item(),
            }
        )

        # Log the metrics
        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(
                    f"train/{key}",
                    value,
                    global_step=self.state.global_step,
                )

        return (total_loss, student_output) if return_outputs else total_loss

    def close(self):
        self.tb_writer.close()
