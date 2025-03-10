from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.optim.loss import FrobeniusLoss
from distil_xlstm.trainer.arguments import KDArguments
from distil_xlstm.utils import xLSTMCausalLMOutput


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
        self.frobenius_criterion = FrobeniusLoss()
        self._teacher_num_attention_layers = len(teacher_model.model.layers)

        self.tb_writer = SummaryWriter(
            log_dir=args.logging_dir, filename_suffix="manual_logs"
        )

    @torch.no_grad()
    def _teacher_forward(self, inputs) -> CausalLMOutputWithPast:
        output = self.teacher(**inputs, output_hidden_states=True)
        return output

    def compute_loss(self, model: DistilxLSTM, inputs, return_outputs=False, **kwargs):
        student_output: xLSTMCausalLMOutput = model(
            **inputs,
            output_hidden_states=True,
        )

        teacher_output = self._teacher_forward(inputs)

        metrics = dict()
        task_loss = student_output.loss
        task_weight = 1
        total_loss = torch.tensor(
            0.0,
            device=task_loss.device,
            dtype=task_loss.dtype,
        )

        is_log_step = self.state.global_step % self.args.logging_steps == 0

        # # Compute KL divergence loss
        if self.args.compute_kl_loss:
            student_logits = rearrange(student_output.logits, "b s d -> (b s) d")
            teacher_logits = rearrange(
                teacher_output.logits.detach(), "b s d -> (b s) d"
            )

            T = self.args.temperature
            scaled_temperature = T**2

            student_probs = F.log_softmax(student_logits / T, dim=-1)
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            kl_loss = self.kl_loss_fn(input=student_probs, target=teacher_probs)
            kl_loss_term = self.args.kl_weight * scaled_temperature * kl_loss

            total_loss += kl_loss_term
            task_weight -= self.args.kl_weight

            metrics.update(
                {
                    "kl_loss": kl_loss.item(),
                    "kl_weight": self.args.kl_weight,
                }
            )

        # # Compute Frobenius loss
        if self.args.compute_frobenius_loss:
            student_h = (
                student_output.hidden_states_per_block
                if self.args.frobenius_norm_computation == "ratio"
                else student_output.hidden_states
            )

            frobenius_loss, norm_per_block = self.frobenius_criterion(
                # skip the first hidden state because the "transformers" library
                # sets the embedding layer's output as the first element in the
                # all_hidden_states tuple
                teacher_hidden_states=teacher_output.hidden_states[1:],
                student_hidden_states=student_h,
                computation=self.args.frobenius_norm_computation,
            )

            total_loss += frobenius_loss * self.args.frobenius_weight
            if self.args.additive_frobenius_weight:
                task_weight -= self.args.frobenius_weight

            metrics.update(
                {
                    "frobenius_loss": frobenius_loss.item(),
                    "frobenius_weight": self.args.frobenius_weight,
                }
            )

            if norm_per_block is not None:
                norm_dict = {
                    f"frobenius_norm/block_{idx}": norm.item()
                    for idx, norm in enumerate(norm_per_block)
                }

                # Log the Frobenius norm per block
                if is_log_step:
                    for key, value in norm_dict.items():
                        self.tb_writer.add_scalar(
                            key,
                            value,
                            global_step=self.state.global_step,
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
        if is_log_step:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(
                    f"train/{key}",
                    value,
                    global_step=self.state.global_step,
                )

        return (total_loss, student_output) if return_outputs else total_loss

    def close(self):
        self.tb_writer.close()
