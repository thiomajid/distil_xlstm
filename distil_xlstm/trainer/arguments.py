from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from distil_xlstm.optim.loss import AlignmentLoss, FrobeniusNormReduction

from ..optim.scheduler import ParamScheduleType


@dataclass
class KDArguments(TrainingArguments):
    train_dataset_url: str = field(default="allenai/c4")
    train_subset: Optional[str] = field(default=None)
    train_split: str = field(default="train")
    train_samples: int = field(default=10_000)

    eval_dataset_url: str = field(default="allenai/c4")
    eval_subset: Optional[str] = field(default=None)
    eval_split: str = field(default="validation")
    eval_samples: int = field(default=5_000)

    teacher_name: str = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={"help": "Name of the model used as teacher model on Hugging Face"},
    )

    quantize_teacher: bool = field(default=True)
    trust_remote_code: bool = field(default=True)

    features: list[str] = field(
        default_factory=list,
        metadata={
            "help": "Columns from the dataset that will be used to create input data for the model"
        },
    )

    delta: float = field(
        default=0.01,
        metadata={
            "help": "$\Delta$ rescales the temperature and alpha at the start of each epoch"
        },
    )

    ce_weight: float = field(
        default=0.4,
        metadata={"help": "Weight of the cross-entropy loss"},
    )

    final_ce_weight: float = field(default=0.4)
    ce_schedule: ParamScheduleType = "decreasing"

    compute_kl_loss: bool = field(default=True)
    kl_weight: float = field(
        default=0.2,
        metadata={"help": "Weight of the KL divergence loss"},
    )

    final_kl_weight: float = field(default=0.1)
    kl_schedule: ParamScheduleType = "decreasing"

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

    compute_alignment_loss: bool = field(
        default=True,
        metadata={"help": "Whether to use the Frobenius norm loss during training"},
    )

    alignment_loss: AlignmentLoss = field(default="cosine")

    alignment_weight: float = field(
        default=0.4,
        metadata={"help": "Weight of the Frobenius norm loss"},
    )

    final_alignment_weight: float = field(default=0.5)
    alignment_schedule: ParamScheduleType = "decreasing"

    frobenius_norm_reduction: FrobeniusNormReduction = field(
        default="block_wise",
        metadata={
            "help": "Method of computing the Frobenius norm: 'average' for average loss, 'ratio' for ratio-based loss"
        },
    )

    additive_alignment_weight: bool = field(
        default=False,
        metadata={
            "help": "Indicate if the Frobenius norm weight combined with the two other weights should sum to 1.0"
        },
    )

    use_dataset_cache: bool = field(
        default=True,
        metadata={
            "help": "Whether to use the dataset cache. If False, the dataset will be reloaded every time."
        },
    )

    dataset_cache_dir: str = field(
        default="./.dataset_cache",
        metadata={
            "help": "Directory where the dataset cache will be stored. If not set, the cache will be stored in the current directory."
        },
    )


class DualGpuTraingArgs(KDArguments):
    student_device: int = field(default=0)
    teacher_device: int = field(default=1)
