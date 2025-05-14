#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from datasets import load_dataset, concatenate_datasets

from distil_xlstm.data import get_dataset


@dataclass
class OfflineDistillationConfig:
    """Configuration for offline distillation dataset creation."""

    # Teacher model config
    teacher_name: str
    quantize_teacher: bool = True
    trust_remote_code: bool = True
    
    # Source dataset config
    dataset_url: str
    data_subset: Optional[str] = None
    data_split: str = "train"
    text_column: str = "text"
    max_seq_length: int = 2048
    num_samples: Union[int, Literal["all"]] = 10_000
    
    # Processing config
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda"
    fp16: bool = True
    
    # Output dataset config
    output_dataset_name: str
    output_dataset_description: str
    push_to_hub: bool = True
    hub_token: Optional[str] = None
    local_dir: Optional[Path] = None
    
    # Cache config
    use_dataset_cache: bool = True
    dataset_cache_dir: str = "./.dataset_cache"


def extract_teacher_outputs(
    config: OfflineDistillationConfig,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    logger:logging.Logger = None,
):
    """Process the dataset and extract teacher outputs."""
    
    logger = logging.getLogger(__name__) if logger is None else logger
    
    # Load the source dataset
    logger.info(f"Loading dataset from {config.dataset_url}")
    source_dataset = get_dataset(
        hub_url=config.dataset_url,
        subset=config.data_subset,
        features=[config.text_column],
        max_seq_length=config.max_seq_length,
        tokenizer=tokenizer,
        split=config.data_split,
        num_samples=config.num_samples,
        token=config.hub_token,
        use_cache=config.use_dataset_cache,
        cache_dir=config.dataset_cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Prepare for PyTorch
    source_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])
    
    # Create a PyTorch DataLoader
    dataloader = torch.utils.data.DataLoader(
        source_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    all_samples = []
    
    # Process the dataset
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing dataset"):
            # Move batch to the right device
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            
            # Forward pass with the teacher model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Extract the outputs
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            
            # Convert to numpy arrays
            logits_np = logits.cpu().numpy()
            
            # Process hidden states
            hidden_states_np = []
            for hidden_state in hidden_states:
                if hidden_state is not None:
                    hidden_states_np.append(hidden_state.cpu().numpy())
            
            # For each item in the batch, create a sample
            for i in range(input_ids.size(0)):
                sample = {
                    "input_ids": input_ids[i].cpu().numpy(),
                    "attention_mask": attention_mask[i].cpu().numpy(),
                    "teacher_logits": logits_np[i],
                }

                # Add hidden states
                sample["hidden_states"] = hidden_states.cpu().numpy()
                
                # for j, hidden_state in enumerate(hidden_states_np):
                #     sample[f"teacher_hidden_state_{j}"] = hidden_state[i]
                
                all_samples.append(sample)
    
    # Create the distillation dataset
    distillation_dataset = HFDataset.from_dict({
        k: [s[k] for s in all_samples] for k in all_samples[0].keys()
    })
    
    return distillation_dataset


@hydra.main(config_path="../configs", config_name="distillation_dataset")
def main(cfg: DictConfig):
    """Main function to create distillation dataset."""
    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting distillation dataset creation...")
    
    # Parse config
    config = OfflineDistillationConfig(**OmegaConf.to_container(cfg, resolve=True))
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer from {config.teacher_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.teacher_name,
        token=config.hub_token,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Padding token set to EOS token.")
    
    # Initialize teacher model
    logger.info(f"Loading teacher model {config.teacher_name}")
    
    # Configure quantization if needed
    quantization_config = None
    if config.quantize_teacher:
        logger.info("Using quantization for the teacher model")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        config.teacher_name,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        token=config.hub_token,
        trust_remote_code=config.trust_remote_code,
        device_map="auto",
        quantization_config=quantization_config if config.quantize_teacher else None,
    )
    
    # Extract teacher outputs and create dataset
    distillation_dataset = extract_teacher_outputs(
        config=config,
        model=model,
        tokenizer=tokenizer,
        logger=logger,
    )
    
    # Save the dataset locally if specified
    if config.local_dir:
        logger.info(f"Saving dataset locally to {config.local_dir}")
        Path(config.local_dir).mkdir(parents=True, exist_ok=True)
        distillation_dataset.save_to_disk(config.local_dir)
    
    # Push to the Hub if specified
    if config.push_to_hub:
        logger.info(f"Pushing dataset to the Hugging Face Hub as {config.output_dataset_name}")
        distillation_dataset.push_to_hub(
            repo_id=config.output_dataset_name,
            token=config.hub_token,
            private=False,
            commit_message="Upload distillation dataset",
        )
    
    logger.info("Distillation dataset creation completed successfully!")
    return distillation_dataset


if __name__ == "__main__":
    main()
