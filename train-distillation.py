from pprint import pprint

from omegaconf import OmegaConf
from transformers import HfArgumentParser
from xlstm import xLSTMLMModel, xLSTMLMModelConfig

from distil_xlstm.trainer import KDArguments

parser = HfArgumentParser(KDArguments)
# model_cfg_parser = HfArgumentParser(xLSTMLMModelConfig)


args: KDArguments = parser.parse_yaml_file(yaml_file="./distillation_config.yaml")[0]
xlstm_cfg: xLSTMLMModelConfig = OmegaConf.load("./xlstm_config.yaml")
model = xLSTMLMModel(xlstm_cfg)
pprint(xlstm_cfg)
