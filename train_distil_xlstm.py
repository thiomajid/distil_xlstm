import logging
from pprint import pprint
from typing import cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)

from distil_xlstm import DistilxLSTMForCausalLM
from distil_xlstm.config import DistilxLSTMConfig
from distil_xlstm.data import get_dataset
from distil_xlstm.optim import AnnealingCallback, ScalarAnnealingScheduler
from distil_xlstm.trainer import KDArguments, KDTrainer
from distil_xlstm.utils import (
    count_parameters,
    count_trainable_parameters,
    next_multiple_of,
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

    logger.info("Trainer arguments:")
    pprint(args)

    logger.info("Loading teacher model and tokenizer...")
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

    # Model instances
    logger.info("Loading teacher model...")
    if args.quantize_teacher:
        logger.warning(
            "quantize_teacher is set to True. The teacher model will be quantized."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_name,
        torch_dtype=torch.float32,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quantization_config if args.quantize_teacher else None,
    )

    # freezing the teacher
    for param in teacher_model.parameters():
        param.requires_grad_(False)

    logger.info(f"Teacher model total parameters: {count_parameters(teacher_model)}")
    logger.info(
        f"Teacher model trainable parameters: {count_trainable_parameters(teacher_model)}"
    )

    # check if the teacher model is not on cuda then set it on cuda
    if not next(teacher_model.parameters()).is_cuda:
        teacher_model = teacher_model.to("cuda")

    logger.info("Creating student model...")
    config_dict = OmegaConf.to_container(cfg["model"], resolve=True)
    xlstm_dict = config_dict.pop("xlstm_config", None)

    teacher_config = teacher_model.config
    xlstm_dict["vocab_size"] = teacher_config.vocab_size
    xlstm_dict["embedding_dim"] = teacher_config.hidden_size

    if config_dict["num_blocks_init"] == "same":
        xlstm_dict["num_blocks"] = teacher_config.num_hidden_layers
    elif config_dict["num_blocks_init"] == "half":
        xlstm_dict["num_blocks"] = teacher_config.num_hidden_layers // 2
    elif config_dict["num_blocks_init"] == "custom":
        pass
    else:
        raise ValueError(
            f"num_blocks_init should be one of ['same', 'half', 'custom'], but got {config_dict['num_blocks_init']}"
        )

    rounded_teacher_num_heads = next_multiple_of(
        teacher_config.num_attention_heads, multiple=4
    )

    # Having too many heads is not a problem per se, but it makes the hidden dimensions
    # too small but also increases the number of computational units
    # dividing by 4 is a good trade-off
    num_heads = rounded_teacher_num_heads // 4
    xlstm_dict["mlstm_block"]["mlstm"]["num_heads"] = num_heads
    xlstm_dict["slstm_block"]["slstm"]["num_heads"] = num_heads
    config_dict["pad_token_id"] = tokenizer.pad_token_id

    config = DistilxLSTMConfig(
        xlstm_config=parse_xlstm_config_dict(xlstm_dict),
        **config_dict,
    )

    logger.info("Model configuration:")
    pprint(config.to_dict())

    student_model = (
        DistilxLSTMForCausalLM.init_for_distillation_with_freezed_head_and_embedding(
            config=config,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
        )
    )
    pprint(student_model)

    logger.info(f"Student model total parameters: {count_parameters(student_model)}")

    logger.info(
        f"Student model trainable parameters: {count_trainable_parameters(student_model)}"
    )

    logger.info(
        f"Loading training dataset from {args.dataset_url} with {args.train_samples} samples"
    )

    train_dataset = get_dataset(
        hub_url=args.dataset_url,
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
        f"Loading evaluation dataset from {args.dataset_url} with {args.eval_samples} samples"
    )

    eval_dataset = get_dataset(
        hub_url=args.dataset_url,
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

    logger.info("Initializing KD trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    annealing_callback = AnnealingCallback(schedulers={})

    if args.temperature_schedule == "decreasing":
        temperature_scheduler = ScalarAnnealingScheduler(
            initial_value=args.temperature,
            final_value=args.final_temperature,
            delta=args.delta,
            schedule_fn_variant="logarithmic",
        )
        annealing_callback.register_scheduler("temperature", temperature_scheduler)

    if args.alignment_schedule == "decreasing":
        alignment_weight_scheduler = ScalarAnnealingScheduler(
            initial_value=args.alignment_weight,
            final_value=args.final_alignment_weight,
            delta=args.delta,
            schedule_fn_variant="logarithmic",
        )

        annealing_callback.register_scheduler(
            "alignment_weight", alignment_weight_scheduler
        )
    if args.kl_schedule == "decreasing":
        kl_weight_scheduler = ScalarAnnealingScheduler(
            initial_value=args.kl_weight,
            final_value=args.final_kl_weight,
            delta=args.delta,
            schedule_fn_variant="logarithmic",
        )
        annealing_callback.register_scheduler("kl_weight", kl_weight_scheduler)

    # Initialize KD trainer
    trainer = KDTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[annealing_callback],
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the trained model
    trainer.save_model()
    trainer.close()

    trainer.push_to_hub()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
