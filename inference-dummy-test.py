from typing import cast

import torch
import yaml
from transformers import AutoTokenizer, HfArgumentParser

from distil_xlstm import DistilxLSTM, DistilxLSTMConfig
from distil_xlstm.trainer import KDArguments
from distil_xlstm.utils import count_parameters

parser = HfArgumentParser(KDArguments)


trainer_args = parser.parse_yaml_file(yaml_file="./distillation_config.yaml")[0]
trainer_args = cast(KDArguments, trainer_args)

with open(trainer_args.xlstm_config_path, "r") as file:
    xlstm_config_dict = yaml.safe_load(file)

xlstm_cfg = DistilxLSTMConfig.parse_xlstm_config_dict(xlstm_config_dict)
cfg = DistilxLSTMConfig(xlstm_cfg=xlstm_cfg)
model = DistilxLSTM(config=cfg)
print(count_parameters(model))

max_length = 10
temperature = 0.7
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Transformers are"
encoding = tokenizer(text, return_tensors="pt", max_length=max_length)
input_ids = encoding["input_ids"]

print(f"Input IDs: {input_ids.tolist()}")

for step in range(max_length):
    logits: torch.Tensor = model(input_ids).logits
    probs = torch.softmax(logits[:, -1, :], dim=-1) / temperature
    max_token = torch.multinomial(probs, num_samples=1)

    input_ids = torch.cat([input_ids, max_token], dim=-1)
    print(tokenizer.decode(max_token.squeeze().item()))

print(tokenizer.decode(input_ids))
