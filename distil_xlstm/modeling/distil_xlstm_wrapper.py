import torch
from torch import nn
from transformers import AutoModelForCausalLM
from xlstm import mLSTMLayer, xLSTMLMModel

from distil_xlstm.config import DistilxLSTMConfig


class DistilxLSTMWrapper(nn.Module):
    def __init__(
        self,
        config: DistilxLSTMConfig,
        teacher_lm: AutoModelForCausalLM,
    ) -> None:
        super().__init__()

        self.config = config
        self.teacher_model = teacher_lm
        self.xlstm = xLSTMLMModel(config.stack_config)

    def __initialize_mlstm_blocks(self):
        teacher = self.teacher_model

        for layer_idx in range(len(self.config.stack_config.num_blocks)):
            if layer_idx in self.config.attention_layers:
                msltm: mLSTMLayer = self.xlstm.xlstm_block_stack.blocks[layer_idx]

                # load query weights
                msltm.q_proj.load_state_dict()

                # load key weights
                msltm.k_proj.load_state_dict()

                # load value weights
                msltm.v_proj.load_state_dict()

                # load out_proj weights
                msltm.proj_down.load_state_dict()

    def forward(self, x: torch.Tensor):
        pass
