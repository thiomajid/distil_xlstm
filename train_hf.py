import argparse
import json

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from distil_xlstm.data import get_cached_dataset
from distil_xlstm.utils import count_parameters


def register_args():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train a model from scratch based on a HuggingFace model architecture"
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID to use for architecture",
    )

    parser.add_argument(
        "--config-overrides",
        type=str,
        default="{}",
        help="JSON string with config overrides (e.g., '{\"hidden_size\": 512}')",
    )

    parser.add_argument(
        "--dataset-url",
        type=str,
        required=True,
        help="URL or name of the dataset on HuggingFace",
    )

    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Subset of the dataset (optional)",
    )

    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Dataset split to use for training",
    )

    parser.add_argument(
        "--eval-split",
        type=str,
        default="validation",
        help="Dataset split to use for evaluation",
    )

    parser.add_argument(
        "--train-samples",
        type=str,
        default="all",
        help="Number of samples for training (integer or 'all')",
    )

    parser.add_argument(
        "--eval-samples",
        type=str,
        default="all",
        help="Number of samples for evaluation (integer or 'all')",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=["text"],
        help="Feature column(s) to use from dataset",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        required=True,
        help="Max sequence length for tokenization",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_model",
        help="Directory to save the model",
    )

    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )

    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for accessing private models/datasets",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push model to Hugging Face Hub after training",
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Model ID for pushing to Hugging Face Hub",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )

    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Disable mixed precision training",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_hf",
        choices=[
            "adamw_hf",
            "adamw_torch",
            "adamw_torch_fused",
            "adamw_apex_fused",
            "adam",
            "adamw_bnb_8bit",
            "sgd",
            "adafactor",
        ],
        help="Optimizer to use for training",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = register_args()

    # Load model config and apply overrides
    config_overrides = json.loads(args.config_overrides)
    print(
        f"Loading model config from {args.model_id} with overrides: {config_overrides}"
    )
    config = AutoConfig.from_pretrained(
        args.model_id, token=args.hf_token, **config_overrides
    )

    print(f"Model config:\n{config}")

    # Initialize tokenizer from the same model
    print(f"Loading tokenizer from {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token)

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Padding token set to EOS token.")

    # Initialize model with random weights
    print(f"Initializing model from scratch with config from {args.model_id}")
    model = AutoModelForCausalLM.from_config(config)
    print(f"Model initialized with {count_parameters(model)}")

    # Load datasets
    print(f"Loading training dataset from {args.dataset_url}")
    train_samples = (
        int(args.train_samples) if args.train_samples != "all" else args.train_samples
    )
    eval_samples = (
        int(args.eval_samples) if args.eval_samples != "all" else args.eval_samples
    )

    train_dataset = get_cached_dataset(
        hub_url=args.dataset_url,
        subset=args.dataset_subset,
        features=args.features,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        split=args.train_split,
        n_samples=train_samples,
        token=args.hf_token,
    )

    eval_dataset = get_cached_dataset(
        hub_url=args.dataset_url,
        subset=args.dataset_subset,
        features=args.features,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        split=args.eval_split,
        n_samples=eval_samples,
        token=args.hf_token,
    )

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])
    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        optim=args.optimizer,  # Add the optimizer parameter here
        load_best_model_at_end=True,
        save_total_limit=2,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hf_token,
        lr_scheduler_type="cosine",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the trained model
    trainer.save_model()

    if args.push_to_hub:
        trainer.push_to_hub()

    print("Training completed successfully!")
