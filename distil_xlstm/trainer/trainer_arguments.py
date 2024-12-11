from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from ..optim.scheduler import ParamScheduleType


@dataclass
class KDArguments(TrainingArguments):
    xlstm_config_path: str = field(default="./xlstm_config.yaml")

    dataset_url: str = field(default="allenai/c4")

    data_subset: Optional[str] = field(default=None)

    train_samples: int = field(default=10_000)
    eval_samples: int = field(default=5_000)

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

    ce_weight: float = field(
        default=1,
        metadata={"help": "Ratio of CE loss"},
    )

    final_ce_weight: float = field(
        default=1,
        metadata={
            "help": "The maximum/minimum value that the ce_weight can take during training"
        },
    )

    ce_schedule: ParamScheduleType = "no-op"

    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Ratio o KL loss"},
    )

    final_kl_weight: float = field(
        default=1,
        metadata={
            "help": "The maximum/minimum value that the kl_weight can take during training"
        },
    )

    kl_schedule: ParamScheduleType = "increase"

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

    temperature_schedule: ParamScheduleType = "decrease"
