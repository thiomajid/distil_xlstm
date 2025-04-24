import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

import torch
from torch import nn
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMLMModelConfig,
)


class DistilxLSTMCausalLMOutput(TypedDict):
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple[torch.Tensor]] = None
    last_hidden_state: Optional[torch.Tensor] = None


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "billions": total_params / 1e9,
        "millions": total_params / 1e6,
    }


def count_trainable_parameters(model: nn.Module, precision: int = 2):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "millions": round(trainable_params / 1e6, precision),
        "billions": round(trainable_params / 1e9, precision),
    }


def download_safetensors(url: str, out_file: str = "./model.safetensors") -> bool:
    """
    Downloads a file from the specified URL and saves it to the given output file path using the curl command.


    Args:
    ---
        url (str): The URL of the file to download.
        out_file (str, optional): The path where the downloaded file will be saved. Defaults to "./model.safetensors".

    Returns:
    ---
        bool: True if the download was successful, False otherwise.

    Raises:
    ---
        subprocess.CalledProcessError: If the curl command fails.
        FileNotFoundError: If the curl command is not found on the system.
    """

    try:
        # Run curl command with progress bar
        _ = subprocess.run(
            [
                "curl",
                "-L",  # Follow redirects
                "-o",
                out_file,  # Output file
                "--progress-bar",  # Show progress
                url,
            ],
            check=True,
        )

        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return False
    except FileNotFoundError:
        print("curl command not found")
        return False


def next_multiple_of(x: int, multiple: int) -> int:
    """
    Returns the next multiple of `multiple` that is greater than or equal to `x`.

    Args:
    ---
        x (int): The number to round up.
        multiple (int): The multiple to round up to.

    Returns:
    ---
        int: The next multiple of `multiple` that is greater than or equal to `x`.
    """
    return ((x + multiple - 1) // multiple) * multiple


def parse_xlstm_config_dict(config_dict: dict[str, Any]):
    # mLSTM block config deserialization
    mlstm_block_dict: dict[str, Any] = config_dict.pop("mlstm_block", None)
    mlstm_block = None
    if mlstm_block_dict:
        mlstm_block = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(**mlstm_block_dict.pop("mlstm")),
            **mlstm_block_dict,
        )

    # sLSTM block config deserialization
    slstm_block_dict: dict[str, Any] = config_dict.pop("slstm_block", None)
    slstm_block = None

    if slstm_block_dict:
        feedforward_dict = slstm_block_dict.pop("feedforward")
        feedforward_config = FeedForwardConfig(**feedforward_dict)
        slstm_block = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(**slstm_block_dict.pop("slstm")),
            feedforward=feedforward_config,
            **slstm_block_dict,
        )

    # xLSTM stack config deserialization
    xlstm_config = xLSTMLMModelConfig(
        mlstm_block=mlstm_block,
        slstm_block=slstm_block,
        **config_dict,
    )

    return xlstm_config


_EvalModelKind = Literal["distil_xlstm", "xlstm", "hub_model"]
PathLike = str | Path


@dataclass
class PerplexityEvaluationConfig:
    def __init__(
        self,
        model_type: _EvalModelKind,
        hub_url: str,
        dataset_url: str,
        data_split: str,
        text_column: str,
        batch_size: int,
        max_seq_length: int,
        num_workers: int,
        local_dir: PathLike | None = None,
        data_subset: Optional[str] = None,
        samples: int | Literal["all"] = "all",
        device: str = "cuda",
        pin_memory: bool = True,
        fp16: bool = True,
        hub_token: Optional[str] = None,
    ):
        self.model_type = model_type
        self.local_dir = local_dir
        self.hub_url = hub_url
        self.dataset_url = dataset_url
        self.data_split = data_split
        self.data_subset = data_subset
        self.text_column = text_column
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.samples = samples
        self.device = device
        self.pin_memory = pin_memory
        self.fp16 = fp16
        self.hub_token = hub_token

    def __repr__(self):
        return f"PerplexityEvaluationConfig({self.__dict__})"

    def __str__(self):
        return self.__repr__()
