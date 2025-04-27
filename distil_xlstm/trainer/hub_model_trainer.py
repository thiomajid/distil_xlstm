import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from distil_xlstm.trainer.arguments import KDArguments


class HubModelTrainer(Trainer):
    def __init__(
        self,
        args: KDArguments,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        is_moe_model: bool = False,
        **kwargs,
    ):
        super().__init__(model=model, args=args, processing_class=tokenizer, **kwargs)
        self.model = model
        self.args = args
        self.is_moe_model = is_moe_model

        self.tb_writer = SummaryWriter(
            log_dir=args.logging_dir, filename_suffix="manual_logs"
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs: CausalLMOutputWithPast | None = None

        outputs = model(**inputs, **kwargs)

        if self.is_moe_model:
            outputs = model(**inputs, **kwargs, output_router_logits=True)

        loss = outputs.loss

        perplexity = torch.exp(loss).item()
        metrics = {
            "train/ce_loss": loss.item(),
            "train/perplexity": perplexity,
        }

        # if self.is_moe_model:
        #     total_z_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        #     for router_logits in outputs.router_logits:
        #         total_z_loss += router_z_loss(router_logits)
        #     z_loss = total_z_loss.mean()

        #     metrics["train/z_loss"] = z_loss.item()
        #     metrics["train/aux_loss"] = outputs.aux_loss.item()

        #     loss = (
        #         loss
        #         + z_loss * self.args.router_loss_coef
        #         + outputs.aux_loss * self.args.load_balancing_loss_coef
        #     )

        metrics["train/total_loss"] = loss.item()
        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.state.global_step)

        if return_outputs:
            return loss, outputs
        else:
            return loss
