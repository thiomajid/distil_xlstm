import argparse
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)

from distil_xlstm import DistilxLSTM
from distil_xlstm.data import get_cached_dataset
from distil_xlstm.optim import AnnealingCallback, ScalarAnnealingScheduler
from distil_xlstm.trainer import KDArguments, KDTrainer
from distil_xlstm.utils import count_parameters, count_trainable_parameters


def register_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trainer-config",
        type=str,
        default="./trainer_arguments.yaml",
        help="Path to the trainer config file",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for pushing the model",
    )

    parser.add_argument(
        "--hub-model-id",
        type=str,
        default=None,
        help="Model ID for the Hugging Face Hub",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Max sequence length for tokenization",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from the Hub",
    )

    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=None,
        help="Override the number of training epochs",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision for evaluation",
    )

    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Disable mixed precision for evaluation",
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )

    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Override batch size for training",
    )

    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help="Override batch size for evaluation",
    )

    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help="Override optimizer (e.g., adamw_torch, adamw_hf)",
    )

    parser.add_argument(
        "--dataset-url",
        type=str,
        required=True,
        help="URL or name of the dataset on HuggingFace",
    )

    parser.add_argument(
        "--train-data-subset",
        type=str,
        default=None,
        help="Subset of the dataset (optional)",
    )

    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Dataset split to use for evaluation",
    )

    parser.add_argument(
        "--train-samples",
        type=str,
        default="all",
        help="Number of samples to evaluate (integer or 'all')",
    )

    parser.add_argument(
        "--eval-data-subset",
        type=str,
        default=None,
        help="Subset of the dataset (optional)",
    )

    parser.add_argument(
        "--eval-split",
        type=str,
        default="validation",
        help="Dataset split to use for evaluation",
    )

    parser.add_argument(
        "--eval-samples",
        type=str,
        default="all",
        help="Number of samples to evaluate (integer or 'all')",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=["text"],
        help="Feature column(s) to use from dataset",
    )

    # Schedule parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Initial temperature for KL distillation",
    )

    parser.add_argument(
        "--final-temperature",
        type=float,
        default=None,
        help="Final temperature for KL distillation",
    )

    parser.add_argument(
        "--temperature-schedule",
        type=str,
        choices=["decreasing", "increasing", "constant"],
        default=None,
        help="Schedule type for temperature annealing",
    )

    parser.add_argument(
        "--kl-weight",
        type=float,
        default=None,
        help="Initial weight for KL loss term",
    )

    parser.add_argument(
        "--final-kl-weight",
        type=float,
        default=None,
        help="Final weight for KL loss term",
    )

    parser.add_argument(
        "--kl-schedule",
        type=str,
        choices=["decreasing", "increasing", "constant"],
        default=None,
        help="Schedule type for KL weight annealing",
    )

    parser.add_argument(
        "--frobenius-weight",
        type=float,
        default=None,
        help="Initial weight for Frobenius loss term",
    )

    parser.add_argument(
        "--final-frobenius-weight",
        type=float,
        default=None,
        help="Final weight for Frobenius loss term",
    )

    parser.add_argument(
        "--frobenius-schedule",
        type=str,
        choices=["decreasing", "increasing", "constant"],
        default=None,
        help="Schedule type for Frobenius weight annealing",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Delta parameter for annealing schedules",
    )

    # Loss computation flags
    parser.add_argument(
        "--compute-kl-loss",
        action="store_true",
        help="Whether to compute KL divergence loss",
    )

    parser.add_argument(
        "--no-compute-kl-loss",
        action="store_false",
        dest="compute_kl_loss",
        help="Whether to skip KL divergence loss",
    )

    parser.add_argument(
        "--compute-frobenius-loss",
        action="store_true",
        help="Whether to compute Frobenius norm loss",
    )

    parser.add_argument(
        "--no-compute-frobenius-loss",
        action="store_false",
        dest="compute_frobenius_loss",
        help="Whether to skip Frobenius norm loss",
    )

    # Student model configuration
    parser.add_argument(
        "--slstm-pos",
        type=int,
        nargs="*",
        default=None,
        help="Positions of SLSTM blocks in student model",
    )

    parser.add_argument(
        "--frobenius-norm-computation",
        type=str,
        choices=["ratio", "standard"],
        default=None,
        help="Method for computing Frobenius norm",
    )

    # New arguments for learning rate, scheduler, teacher model and quantization
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate for training",
    )

    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default=None,
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
        help="Override learning rate scheduler type",
    )

    parser.add_argument(
        "--teacher-name",
        type=str,
        default=None,
        help="Override teacher model name/path",
    )

    parser.add_argument(
        "--quantize-teacher",
        action="store_true",
        help="Use quantization for teacher model",
    )

    parser.add_argument(
        "--no-quantize-teacher",
        action="store_false",
        dest="quantize_teacher",
        help="Disable quantization for teacher model",
    )

    # Add new argument for additive_frobenius_weight
    parser.add_argument(
        "--additive-frobenius-weight",
        action="store_true",
        help="Use additive Frobenius weight computation",
    )

    parser.add_argument(
        "--no-additive-frobenius-weight",
        action="store_false",
        dest="additive_frobenius_weight",
        help="Use non-additive Frobenius weight computation",
    )

    return parser.parse_args()


def main():
    args = register_args()
    parser = HfArgumentParser(KDArguments)

    # Load trainer arguments from YAML file
    trainer_args = parser.parse_yaml_file(yaml_file=args.trainer_config)[0]
    trainer_args = cast(KDArguments, trainer_args)

    # Override training arguments with command line arguments
    if args.num_train_epochs is not None:
        trainer_args.num_train_epochs = args.num_train_epochs
    if args.fp16:
        trainer_args.fp16 = args.fp16
    if args.gradient_accumulation_steps is not None:
        trainer_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.per_device_train_batch_size is not None:
        trainer_args.per_device_train_batch_size = args.per_device_train_batch_size
    if args.per_device_eval_batch_size is not None:
        trainer_args.per_device_eval_batch_size = args.per_device_eval_batch_size
    if args.optim is not None:
        trainer_args.optim = args.optim

    # Override schedule parameters
    if args.temperature is not None:
        trainer_args.temperature = args.temperature
    if args.final_temperature is not None:
        trainer_args.final_temperature = args.final_temperature
    if args.temperature_schedule is not None:
        trainer_args.temperature_schedule = args.temperature_schedule

    if args.kl_weight is not None:
        trainer_args.kl_weight = args.kl_weight
    if args.final_kl_weight is not None:
        trainer_args.final_kl_weight = args.final_kl_weight
    if args.kl_schedule is not None:
        trainer_args.kl_schedule = args.kl_schedule

    if args.frobenius_weight is not None:
        trainer_args.frobenius_weight = args.frobenius_weight
    if args.final_frobenius_weight is not None:
        trainer_args.final_frobenius_weight = args.final_frobenius_weight
    if args.frobenius_schedule is not None:
        trainer_args.frobenius_schedule = args.frobenius_schedule

    if args.delta is not None:
        trainer_args.delta = args.delta

    # Override loss computation flags
    if hasattr(args, "compute_kl_loss"):
        trainer_args.compute_kl_loss = args.compute_kl_loss
    if hasattr(args, "compute_frobenius_loss"):
        trainer_args.compute_frobenius_loss = args.compute_frobenius_loss
    if hasattr(args, "additive_frobenius_weight"):
        trainer_args.additive_frobenius_weight = args.additive_frobenius_weight

    if args.frobenius_norm_computation is not None:
        trainer_args.frobenius_norm_computation = args.frobenius_norm_computation

    # Override learning rate and scheduler if provided
    if args.learning_rate is not None:
        trainer_args.learning_rate = args.learning_rate

    # if args.lr_scheduler_type is not None:
    #     trainer_args.lr_scheduler_type = args.lr_scheduler_type

    # Override teacher model name if provided
    if args.teacher_name is not None:
        trainer_args.teacher_name = args.teacher_name

    # Override teacher quantization if provided
    if hasattr(args, "quantize_teacher"):
        trainer_args.quantize_teacher = args.quantize_teacher

    # Update hub_model_id if provided
    if args.hub_model_id is not None:
        trainer_args.hub_model_id = args.hub_model_id

    # Update dataset-related args
    trainer_args.dataset_url = args.dataset_url
    if args.train_data_subset is not None:
        trainer_args.train_subset = args.train_data_subset
    if args.train_split is not None:
        trainer_args.train_split = args.train_split
    if args.train_samples != "all":
        trainer_args.train_samples = int(args.train_samples)

    if args.eval_data_subset is not None:
        trainer_args.eval_subset = args.eval_data_subset
    if args.eval_split is not None:
        trainer_args.eval_split = args.eval_split
    if args.eval_samples != "all":
        trainer_args.eval_samples = int(args.eval_samples)

    if args.features is not None:
        trainer_args.features = args.features

    if args.hf_token is not None:
        trainer_args.hub_token = args.hf_token
    else:
        raise ValueError(
            "Hugging Face token is required for pushing the model to the Hub."
        )

    print("Loading teacher model and tokenizer...")
    # Load teacher model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        trainer_args.teacher_name,
        token=trainer_args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Padding token set to EOS token.")

    # Determine max sequence length
    if args.max_seq_length is None:
        raise ValueError("Max sequence length must be specified for tokenization.")
    if args.max_seq_length > tokenizer.model_max_length:
        raise ValueError(
            f"Max sequence length {args.max_seq_length} exceeds model max length {tokenizer.model_max_length}."
        )

    print(f"Loading training dataset from {trainer_args.dataset_url}...")
    # Load training dataset using caching
    train_dataset = get_cached_dataset(
        hub_url=trainer_args.dataset_url,
        subset=getattr(trainer_args, "train_subset", None),
        features=trainer_args.features,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        split=trainer_args.train_split,
        n_samples=trainer_args.train_samples,
        token=trainer_args.hub_token,
    )

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])

    print("Loading evaluation dataset...")
    # Load evaluation dataset using caching
    eval_dataset = get_cached_dataset(
        hub_url=trainer_args.dataset_url,
        subset=getattr(trainer_args, "eval_subset", None),
        features=trainer_args.features,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        split=trainer_args.eval_split,
        n_samples=trainer_args.eval_samples,
        token=trainer_args.hub_token,
    )

    eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "length"])

    # Model instances
    print("Loading teacher model...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        trainer_args.teacher_name,
        torch_dtype=torch.float32,
        token=trainer_args.hub_token,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config
        if trainer_args.quantize_teacher
        else None,
    )

    # freezing the teacher
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    print(f"Teacher model total parameters: {count_parameters(teacher_model)}")
    print(
        f"Teacher model trainable parameters: {count_trainable_parameters(teacher_model)}"
    )

    # check if the teacher model is not on cuda then set it on cuda
    if not next(teacher_model.parameters()).is_cuda:
        teacher_model = teacher_model.to("cuda")

    print("Creating student model...")
    # Create student model with frozen head and embedding
    student_model = DistilxLSTM.init_for_distillation_with_freezed_head_and_embedding(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        xlstm_config_path=trainer_args.xlstm_config_path,
        max_sequence_length=args.max_seq_length,
        slstm_pos=args.slstm_pos,  # Pass slstm_pos from args to function
        v2=trainer_args.v2_init,
    )

    print(f"Student model total parameters: {count_parameters(student_model)}")

    print(
        f"Student model trainable parameters: {count_trainable_parameters(student_model)}"
    )

    print("Initializing KD trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    annealing_callback = AnnealingCallback(schedulers={})

    if trainer_args.temperature_schedule == "decreasing":
        temperature_scheduler = ScalarAnnealingScheduler(
            initial_value=trainer_args.temperature,
            final_value=trainer_args.final_temperature,
            delta=trainer_args.delta,
            schedule_fn_variant="logarithmic",
        )
        annealing_callback.register_scheduler("temperature", temperature_scheduler)

    if trainer_args.frobenius_schedule == "decreasing":
        frobenius_weight_scheduler = ScalarAnnealingScheduler(
            initial_value=trainer_args.frobenius_weight,
            final_value=trainer_args.final_frobenius_weight,
            delta=trainer_args.delta,
            schedule_fn_variant="logarithmic",
        )
        annealing_callback.register_scheduler(
            "frobenius_weight", frobenius_weight_scheduler
        )
    if trainer_args.kl_schedule == "decreasing":
        kl_weight_scheduler = ScalarAnnealingScheduler(
            initial_value=trainer_args.kl_weight,
            final_value=trainer_args.final_kl_weight,
            delta=trainer_args.delta,
            schedule_fn_variant="logarithmic",
        )
        annealing_callback.register_scheduler("kl_weight", kl_weight_scheduler)

    # Initialize KD trainer
    trainer = KDTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        args=trainer_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[annealing_callback],
    )

    # Train the model
    print("Starting training...")
    trainer.train()
    trainer.log({"num_tokens": trainer.num_tokens(trainer.get_train_dataloader())})

    # Save the trained model
    trainer.save_model()
    trainer.push_to_hub()

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
