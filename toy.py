from typing import cast

import yaml
from transformers import HfArgumentParser

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
