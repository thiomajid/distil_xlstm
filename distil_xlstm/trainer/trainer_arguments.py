from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from ..optim.scheduler import ParamScheduleType


@dataclass
class KDArguments(TrainingArguments):
    xlstm_config_path: str = field(default="./xlstm_config.yaml")

    dataset_url: str = field(default="Salesforce/wikitext")

    data_subset: Optional[str] = field(default=None)

    context_length: int = field(default=4096)

    train_samples: int = field(default=5000)

    teacher_name: str = field(
        default="HuggingFaceH4/zephyr-7b-beta",
        metadata={"help": "Name of the model used as teacher model on Hugging Face"},
    )

    quantize_teacher: bool = field(default=True)

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
