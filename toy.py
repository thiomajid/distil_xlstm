import torch
import yaml

from distil_xlstm import DistilxLSTMConfig, DistilxLSTMForCausalLM
from distil_xlstm.utils import DistilxLSTMCausalLMOutput


def main():
    with open("./configs/model/demo_config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = DistilxLSTMConfig.from_dict(cfg_dict)
    model = DistilxLSTMForCausalLM(config=config)
    print(model)

    dummy_input = torch.randint(
        1, config.xlstm_cfg.vocab_size, (2, config.xlstm_cfg.context_length)
    )

    output: DistilxLSTMCausalLMOutput = model(
        dummy_input,
        output_hidden_states=True,
    )
    print(output["logits"].shape)

    states = torch.cat(output["hidden_states"], dim=0)
    print(states.shape)


if __name__ == "__main__":
    main()
