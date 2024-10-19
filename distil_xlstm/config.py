from dataclasses import dataclass, field

from transformers import TrainingArguments


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

    max_ce_weight: float = field(
        default=1,
        metadata={
            "help": "The maximum value that the ce_weight can take during training"
        },
    )

    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Ratio o KL loss"},
    )

    max_kl_weight: float = field(
        default=1,
        metadata={
            "help": "The maximum value that the kl_weight can take during training"
        },
    )

    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature used to softnen probs"},
    )

    max_temperature: float = field(default=1)
