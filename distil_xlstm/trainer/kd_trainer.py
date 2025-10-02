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
        student_model: DistilxLSTMForCausalLM,
        args: KDArguments,
        tokenizer: AutoTokenizer,
        teacher_model: AutoModelForCausalLM = None,  # Made optional for offline distillation
        **kwargs,
    ) -> None:
        super().__init__(
            args=args,
            model=student_model,
            tokenizer=tokenizer,
            **kwargs,
        )

        self.args = args
        self.teacher = teacher_model  # Can be None for offline distillation

        self.kl_loss_fn = partial(F.kl_div, reduction="batchmean")
        self.alignment_criterion = (
            self._get_alignment_criterion() if args.compute_alignment_loss else None
        )

        self.tb_writer = SummaryWriter(
            log_dir=args.logging_dir, filename_suffix="manual_logs"
        )

        # Flag to determine if we're using offline distillation
        self.use_offline_distillation = teacher_model is None

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
        """Run teacher model forward pass (only used for online distillation)"""
        if self.teacher is None:
            raise ValueError("Teacher model is not available for online distillation")

        output = self.teacher(**inputs, output_hidden_states=True)
        return output

    def compute_loss(
        self,
        model: DistilxLSTMForCausalLM,
        inputs,
        return_outputs=False,
        **kwargs,
    ):
        # Extract teacher outputs if using offline distillation
        teacher_logits = None
        teacher_hidden_states = None

        if self.use_offline_distillation:
            # Extract teacher outputs from the inputs
            if "teacher_logits" in inputs:
                teacher_logits = inputs.pop("teacher_logits")

            if "hidden_states" in inputs:
                teacher_hidden_states = inputs.pop("hidden_states")

            # If these are not present, raise an error
            if teacher_logits is None or teacher_hidden_states is None:
                raise ValueError(
                    "Offline distillation requires 'teacher_logits' and 'hidden_states' "
                    "in the input batch. Make sure you're using a dataset created with "
                    "the offline distillation script."
                )

        # Run the student model
        student_output: DistilxLSTMCausalLMOutput = model(
            **inputs,
            output_hidden_states=True,
        )

        # Get teacher outputs (either from cached dataset or by running the teacher)
        if not self.use_offline_distillation:
            teacher_output = self._teacher_forward(inputs)
            teacher_logits = teacher_output.logits
            teacher_hidden_states = teacher_output.hidden_states

        metrics = dict()
        task_loss = student_output["loss"].mean()
        task_weight = 1
        total_loss = torch.tensor(
            0.0,
            device=task_loss.device,
            dtype=task_loss.dtype,
        )

        # Compute KL divergence loss
        if self.args.compute_kl_loss:
            student_logits = rearrange(student_output["logits"], "b s d -> (b s) d")
            teacher_logits = rearrange(teacher_logits, "b s d -> (b s) d")

            # Ensure teacher logits are on the right device and detached
            if isinstance(teacher_logits, torch.Tensor):
                teacher_logits = teacher_logits.to(student_logits.device).detach()
            else:
                # Handle numpy array case from offline dataset
                teacher_logits = torch.tensor(
                    teacher_logits, device=student_logits.device
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

        # Compute Frobenius/alignment loss
        if self.args.compute_alignment_loss:
            student_hidden_states = student_output[
                "hidden_states"
            ]  # a tuple of tensors
            student_hidden_states = torch.cat(student_hidden_states, dim=0)

            if self.use_offline_distillation:
                # Convert numpy array to tensor if needed
                teacher_hidden_states = torch.tensor(
                    teacher_hidden_states, device=student_hidden_states.device
                )
            else:
                # Online distillation path (from teacher model output)
                if isinstance(teacher_hidden_states, tuple):
                    # Skip the first hidden state (embedding output)
                    teacher_hidden_states = torch.cat(teacher_hidden_states[1:], dim=0)
                else:
                    teacher_hidden_states = teacher_hidden_states[1:]

            # Ensure teacher hidden states are on the right device
            teacher_hidden_states = teacher_hidden_states.to(
                student_hidden_states.device
            )

            # Compute the alignment loss
            alignment_loss = None
            if self.args.alignment_loss == "frobenius":
                alignment_loss = self.alignment_criterion(
                    teacher_hidden_states=teacher_hidden_states,
                    student_hidden_states=student_hidden_states,
                )
            elif self.args.alignment_loss == "cosine":
                # Flatten the tensors from [batch, seq, hidden] to [batch*seq, hidden]
                student_h_flat = rearrange(student_hidden_states, "b s h -> (b s) h")
                teacher_h_flat = rearrange(teacher_hidden_states, "b s h -> (b s) h")

                # Create target tensor with same batch size as flattened tensors
                target = torch.ones(
                    student_h_flat.shape[0], device=student_hidden_states.device
                )

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
