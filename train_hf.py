import logging
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
)

from distil_xlstm.data import get_dataset
from distil_xlstm.optim.callbacks import PerplexityLoggingCallback
from distil_xlstm.trainer.arguments import KDArguments
from distil_xlstm.utils import count_parameters


@hydra.main(config_path="./configs", config_name="train_config", version_base="1.2")
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

    MODEL_ID: str = cfg["hub_model_id"]
    MAX_SEQ_LENGTH = cfg["max_seq_length"]

    config = AutoConfig.from_pretrained(
        MODEL_ID,
        token=args.hub_token,
    )

    for k, v in OmegaConf.to_container(cfg["model"], resolve=True).items():
        if hasattr(config, k):
            setattr(config, k, v)
        else:
            logger.warning(f"Model config does not have attribute {k}. Skipping.")

    logger.info(f"Model config:\n{config}")

    # Initialize tokenizer from the same model
    logger.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=args.hub_token)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Padding token set to EOS token.")

    # Initialize model with random weights
    logger.info(f"Initializing untrained model {MODEL_ID}")

    model = AutoModelForCausalLM.from_config(config)

    logger.info(f"Model initialized with {count_parameters(model)}")

    # raise Exception("steofsd")

    # Load datasets
    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    train_dataset = get_dataset(
        hub_url=args.train_dataset_url,
        subset=args.train_subset,
        features=args.features,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])

    logger.info(
        f"Loading evaluation dataset from {args.eval_dataset_url} with {args.eval_samples} samples"
    )

    eval_dataset = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=args.eval_subset,
        features=args.features,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])

    # Data collator
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
