#!/usr/bin/env python3
"""
Create offline distillation dataset with configurable teacher model.
Extracts teacher logits and hidden states per attention block, saves them as numpy arrays,
and pushes the dataset to the Hugging Face Hub.
"""

import logging
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from distil_xlstm.create_distillation_dataset import main as create_dataset_main


@hydra.main(config_path="./configs", config_name="distillation_dataset")
def main(cfg: DictConfig):
    """Main entry point for the distillation dataset creation script."""
    return create_dataset_main(cfg)


if __name__ == "__main__":
    main()
