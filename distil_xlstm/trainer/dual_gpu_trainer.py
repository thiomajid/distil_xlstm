from functools import partial

import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.optim.loss import FrobeniusLoss
from distil_xlstm.trainer.arguments import KDArguments


class DualGpuTrainer(Trainer):
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
