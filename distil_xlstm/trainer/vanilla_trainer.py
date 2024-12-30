from transformers import AutoTokenizer, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast

from distil_xlstm.modeling import DistilxLSTM
from distil_xlstm.trainer.trainer_arguments import KDArguments


class VanillaTrainer(Trainer):
    def __init__(
        self,
        model: DistilxLSTM,
        args: KDArguments,
        tokenizer: AutoTokenizer,
        **kwargs,
    ) -> None:
        super().__init__(
            args=args,
            model=model,
            processing_class=tokenizer,
            **kwargs,
        )

        self.args = args

    def compute_loss(self, model: DistilxLSTM, inputs, return_outputs=False, **kwargs):
        student_output: CausalLMOutputWithPast = model(
            **inputs,
            output_hidden_states=True,
        )

        if return_outputs:
            return (student_output.loss, student_output)

        return student_output.loss
