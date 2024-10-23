from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMLMModelConfig,
)

from distil_xlstm import DistilxLSTM, DistilxLSTMConfig

CONFIG_MAPPING.register("xlstm", DistilxLSTMConfig)
AutoConfig.register("xlstm", DistilxLSTMConfig)
AutoModelForCausalLM.register(DistilxLSTMConfig, DistilxLSTM)


xlstm_cfg = xLSTMLMModelConfig(
    vocab_size=34_000,
    context_length=1024,
    num_blocks=12,
    embedding_dim=512,
    slstm_at=[1, 3, 5, 7, 9],
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="vanilla",
            num_gates=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
)

config = DistilxLSTMConfig(xlstm_cfg=xlstm_cfg)
model = DistilxLSTM(config)

model.push_to_hub("thiomajid/distil-xlstm")
