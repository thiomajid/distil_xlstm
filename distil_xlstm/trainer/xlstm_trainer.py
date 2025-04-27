import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, Trainer

from distil_xlstm.modeling import DistilxLSTMForCausalLM
from distil_xlstm.trainer.arguments import KDArguments
from distil_xlstm.utils import DistilxLSTMCausalLMOutput


class xLSTMTrainer(Trainer):
    def __init__(
        self,
        args: KDArguments,
        model: DistilxLSTMForCausalLM,
        tokenizer: AutoTokenizer,
        **kwargs,
    ):
        super().__init__(model=model, args=args, processing_class=tokenizer, **kwargs)
        self.model = model
        self.args = args

        self.tb_writer = SummaryWriter(
            log_dir=args.logging_dir, filename_suffix="manual_logs"
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs: DistilxLSTMCausalLMOutput = model(**inputs, **kwargs)
        loss = outputs["loss"]

        perplexity = torch.exp(loss).item()
        metrics = {
            "train/ce_loss": loss.item(),
            "train/perplexity": perplexity,
            "train/total_loss": loss.item(),
        }

        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, self.state.global_step)

        if return_outputs:
            return loss, outputs
        else:
            return loss
