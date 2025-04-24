import subprocess
from typing import Optional, TypedDict

import torch
from torch import nn


class xLSTMCausalLMOutput(TypedDict):
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    hidden_states_per_block: Optional[torch.Tensor] = None


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
