from typing import cast

import torch
import yaml
from transformers import HfArgumentParser

from distil_xlstm import DistilxLSTM, DistilxLSTMConfig
from distil_xlstm.trainer import KDArguments
from distil_xlstm.utils import count_parameters

if __name__ == "__main__":
    parser = HfArgumentParser(KDArguments)

    trainer_args = parser.parse_yaml_file(yaml_file="./trainer_arguments.yaml")[0]
    trainer_args = cast(KDArguments, trainer_args)

    with open("./demo_config.yaml", "r") as file:
        xlstm_config_dict = yaml.safe_load(file)

    xlstm_cfg = DistilxLSTMConfig.parse_xlstm_config_dict(xlstm_config_dict)
    cfg = DistilxLSTMConfig(xlstm_cfg=xlstm_cfg)
    model = DistilxLSTM(config=cfg)

    print(count_parameters(model))

    dummy_input = torch.randint(
        0, 100, (2, xlstm_cfg.context_length), device=model.device
    )

    output = model(dummy_input, frobenius_computation="ratio")

    print(output["logits"].shape)
