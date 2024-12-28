from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from ..optim.scheduler import ParamScheduleType


@dataclass
class KDArguments(TrainingArguments):
    xlstm_config_path: str = field(default="./xlstm_config.yaml")

    dataset_url: str = field(default="allenai/c4")

    train_subset: Optional[str] = field(default=None)
    train_samples: int = field(default=10_000)
    train_split: str = field(default="train")

    eval_subset: Optional[str] = field(default=None)
    eval_samples: int = field(default=5_000)
    eval_split: str = field(default="validation")

    teacher_name: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Name of the model used as teacher model on Hugging Face"},
    )

    quantize_teacher: bool = field(default=True)

    features: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Columns from the dataset that will be used to create input data for the model"
        },
    )

    delta: float = field(
        default=0.09,
        metadata={
            "help": "$\Delta$ rescales the temperature and alpha at the start of each epoch"
        },
    )

    alpha: float = field(
        default=0.8,
        metadata={"help": "$\alpha$ term weighing the loss function terms"},
    )

    final_alpha: float = field(default=0.5)
    alpha_schedule: ParamScheduleType = "decreasing"

    temperature: float = field(
        default=2,
        metadata={"help": "Temperature used to softnen probs"},
    )

    final_temperature: float = field(
        default=1,
        metadata={
            "help": "The maximum/minimum value that the temperature can take during training"
        },
    )

    temperature_schedule: ParamScheduleType = "decreasing"
