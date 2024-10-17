from dataclasses import dataclass
from typing import List, Literal, Optional

from xlstm import xLSTMBlockStackConfig


@dataclass
class DistilxLSTMConfig:
    """
    Configuration object used to initialize an xLSTM model.

    Attributes
    ----------
    stack_config : xLSTMBlockStackConfig
        Configuration object to initialize xLSTM components such as mLSTM and sLSTM.

    attention_layers : Optional[List[int]]
        An optional property that indicates at which positions xLSTM layers initialized
        from attention layers should be inserted.

    init_with_qkvo : bool
        A flag indicating if `mLSTM` blocks should be initialized using attention layers
        from the teacher model.

    teacher_model : str
        The repo ID of a causal LM from Hugging Face that will be used as the teacher model
        during knowledge distillation.

    use_slstm : bool
        A flag indicating whether `sLSTM` blocks should be added to the *sequence mixing* stack.
    """

    stack_config: xLSTMBlockStackConfig
    attention_layers: Optional[List[int]] = None
    init_with_qkvo: bool = False
    teacher_model: str = "HuggingFaceH4/zephyr-7b-beta"
    use_slstm: bool = False

    def __post_init__(self):
        assert (
            self.stack_config is not None
        ), "The xLSTM stack configuration object can not be `None`"

        assert (
            self.teacher_model is not None or len(self.teacher_model) != 0
        ), f"{self.teacher_model} is an invalid for the `teacher_model` attribute"

        if self.init_with_qkvo and self.attention_layers is None:
            raise ValueError(
                "`init_with_qkvo` is set to True, but `attention_layers` is None."
            )

        if (
            self.attention_layers is not None
            and len(self.attention_layers) <= self.stack_config.num_blocks
        ):
            raise ValueError("You can not have more attention layers than xLSTM block")

        # check that we are not trying to initialize an sLSTM block with an attention layer
        if self.use_slstm and self.init_with_qkvo:
            slstm_pos = self.stack_config.slstm_at
            for pos in slstm_pos:
                if pos in self.attention_layers:
                    raise ValueError(
                        f"Invalid position for an sLSTM module at {pos}. sLSTM modules can not be initialized from QKVO matrices"
                    )


@dataclass
class KDTrainerArguments:
    ce_weight: float = 1e-5
    device: Literal["cpu", "cuda"] = "cuda"
    num_devices: int = 1
    num_epochs: int = 1
    batch_size: int = 128
    num_workers: int = 1
    hf_repo: str
    dataset_url: str
    output_dir: str
