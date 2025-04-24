#!/usr/bin/env python3
import logging
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
)

from distil_xlstm.config import DistilxLSTMConfig
from distil_xlstm.data import get_dataset
from distil_xlstm.modeling import DistilxLSTMForCausalLM
from distil_xlstm.optim.callbacks import PerplexityLoggingCallback
from distil_xlstm.trainer.arguments import KDArguments
from distil_xlstm.utils import (
    count_parameters,
    count_trainable_parameters,
    parse_xlstm_config_dict,
)


@hydra.main(config_path="./configs", config_name="train_config")
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    parser = HfArgumentParser(KDArguments)

    # Load trainer arguments from YAML file
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]
    args = cast(KDArguments, args)

    config = DistilxLSTMConfig(
        xlstm_config=parse_xlstm_config_dict(
            OmegaConf.to_container(cfg["model"], resolve=True)
        )
    )

    logger.info("Loading tokenizer...")
    # Load teacher model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_name,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Padding token set to EOS token.")

    config.pad_token_id = tokenizer.pad_token_id

    # Model instance
    logger.info("Creating xLSTM model...")
    model = DistilxLSTMForCausalLM(config)

    logger.info(f"model total parameters: {count_parameters(model)}")
    logger.info(f"model trainable parameters: {count_trainable_parameters(model)}")

    logger.info(
        f"Loading training dataset from {args.train_} with {args.train_samples} samples"
    )

    train_dataset = get_dataset(
        hub_url=args.train_,
        subset=args.train_subset,
        features=args.features,
        max_seq_length=config.xlstm_config.context_length,
        tokenizer=tokenizer,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])

    logger.info(
        f"Loading evaluation dataset from {args.train_} with {args.eval_samples} samples"
    )

    eval_dataset = get_dataset(
        hub_url=args.train_,
        subset=args.eval_subset,
        features=args.features,
        max_seq_length=config.xlstm_config.context_length,
        tokenizer=tokenizer,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])

    logger.info("Initializing trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[PerplexityLoggingCallback()],
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the trained model
    trainer.save_model()

    if args.push_to_hub:
        trainer.push_to_hub()

    logger.info("Training completed successfully!")
