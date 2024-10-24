from typing import cast

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)

from distil_xlstm import DistilxLSTM, DistilxLSTMConfig
from distil_xlstm.data import get_dataset
from distil_xlstm.trainer import KDArguments, KDTrainer


def train_with_distillation():
    parser = HfArgumentParser(KDArguments)
    args = parser.parse_yaml_file(yaml_file="./distillation_config.yaml")[0]
    args = cast(KDArguments, args)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_name)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_data = get_dataset(
        args=args,
        split="train",
        tokenizer=tokenizer,
    )

    with open(args.xlstm_config_path, "r") as file:
        xlstm_config_dict = yaml.safe_load(file)

    xlstm_cfg = DistilxLSTMConfig.parse_xlstm_config_dict(xlstm_config_dict)
    xlstm_cfg.vocab_size = len(tokenizer)
    cfg = DistilxLSTMConfig(xlstm_cfg=xlstm_cfg)
    model = DistilxLSTM(config=cfg)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_name,
        quantization_config=quantization_config if args.quantize_teacher else None,
    )

    trainer = KDTrainer(
        teacher_model=teacher_model,
        model=model,
        args=args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=train_data,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    train_with_distillation()
