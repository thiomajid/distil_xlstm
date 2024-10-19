from dataclasses import dataclass, field

from transformers import TrainingArguments

from ..optim.scheduler import ParamScheduleType


@dataclass
class KDArguments(TrainingArguments):
    teacher_name: str = field(
        default="HuggingFaceH4/zephyr-7b-beta",
        metadata={"help": "Name of the model used as teacher model on Hugging Face"},
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
