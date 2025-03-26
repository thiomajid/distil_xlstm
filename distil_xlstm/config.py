from dataclasses import asdict
from typing import Any, Dict, Optional

from transformers import PretrainedConfig
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMLMModelConfig,
)
from xlstm.xlstm_large import xLSTMLargeConfig


class DistilxLSTMConfig(PretrainedConfig):
    model_type = "xlstm"

    def __init__(
        self,
        use_tfla: bool = False,
        xlstm_cfg: Optional[xLSTMLMModelConfig] = None,
        tfla_config: Optional[xLSTMLargeConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if xlstm_cfg is None:
            xlstm_cfg = xLSTMLMModelConfig()

        if tfla_config is None:
            tfla_config = xLSTMLargeConfig(
                embedding_dim=1, num_heads=1, num_blocks=1, vocab_size=1
            )

        self.xlstm_cfg = xlstm_cfg
        self.use_tfla = use_tfla
        self.tfla_config = tfla_config

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()

        # Making sure that 'xlstm_cfg' is serialized
        output["xlstm_cfg"] = asdict(self.xlstm_cfg)
        output["tfla_config"] = asdict(self.tfla_config) if self.tfla_config else None
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        xlstm_config_dict: Dict[str, any] = config_dict.pop("xlstm_config")
        xlstm_cfg = cls.parse_xlstm_config_dict(xlstm_config_dict)

        tfla_dict = config_dict.pop("tfla_config")
        tfla_config = xLSTMLargeConfig(**tfla_dict) if tfla_dict else None

        return cls(xlstm_cfg=xlstm_cfg, tfla_config=tfla_config, **config_dict)

    @staticmethod
    def parse_xlstm_config_dict(config_dict: Dict[str, any]):
        # mLSTM block config deserialization
        mlstm_block_dict: Dict[str, any] = config_dict.pop("mlstm_block", None)
        mlstm_block = None
        if mlstm_block_dict:
            mlstm_block = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(**mlstm_block_dict.pop("mlstm")),
                **mlstm_block_dict,
            )

        # sLSTM block config deserialization
        slstm_block_dict: Dict[str, any] = config_dict.pop("slstm_block", None)
        slstm_block = None

        if slstm_block_dict:
            feedforward_dict = slstm_block_dict.pop("feedforward")
            feedforward_config = FeedForwardConfig(**feedforward_dict)
            slstm_block = sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(**slstm_block_dict.pop("slstm")),
                feedforward=feedforward_config,
                **slstm_block_dict,
            )

        # xLSTM stack config deserialization
        xlstm_cfg = xLSTMLMModelConfig(
            mlstm_block=mlstm_block,
            slstm_block=slstm_block,
            **config_dict,
        )

        return xlstm_cfg
