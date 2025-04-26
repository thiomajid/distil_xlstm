import argparse
import json
from pathlib import Path

import yaml
from transformers import (
    AutoConfig,
    Gemma2Config,
    JambaConfig,
    LlamaConfig,
    MambaConfig,
    MixtralConfig,
    Qwen2MoeConfig,
)

_registry: dict[str, AutoConfig] = {
    "mistralai/Mixtral-8x7B-v0.1": MixtralConfig,
    "google/gemma-2-2b": Gemma2Config,
    "state-spaces/mamba-390m-hf": MambaConfig,
    "ai21labs/AI21-Jamba-Mini-1.5": JambaConfig,
    "Qwen/Qwen1.5-MoE-A2.7B": Qwen2MoeConfig,
    "HuggingFaceTB/SmolLM2-360M-Instruct": LlamaConfig,
}


def export_config(model_hub_id: str, output_dir: str, token: str = None):
    """
    Export the configuration of a model to a JSON file.

    Args:
        model_name (str): The name of the model.
        output_dir (str): The directory to save the configuration file.
    """

    # Check if the model name is in the registry
    if model_hub_id not in _registry:
        raise ValueError(f"Model {model_hub_id} not found in registry.")

    # Get the configuration class from the registry
    config_class = _registry[model_hub_id]

    # Create a configuration object
    out_dir = Path(output_dir)
    config = config_class.from_pretrained(model_hub_id, token=token)

    config.save_pretrained(out_dir)

    # convert the JSON file to YAML
    with open(f"{out_dir}/config.json", "r") as json_file:
        config_dict = json.load(json_file)

    model_name = model_hub_id.split("/")[-1]
    with open(f"{out_dir}/model/{model_name}.yaml", "w") as yaml_file:
        yaml.dump(config_dict, yaml_file, default_flow_style=False)

    # delete the JSON file
    out_dir.joinpath("config.json").unlink()

    print(f"Configuration for {model_hub_id} saved to {out_dir}/{model_hub_id}.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export model configuration to JSON and YAML."
    )

    parser.add_argument(
        "--model",
        type=str,
        help="The name of the model.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="The directory to save the configuration file.",
    )

    parser.add_argument(
        "--token",
        type=str,
        help="The token to use for authentication.",
    )

    args = parser.parse_args()

    export_config(model_hub_id=args.model, output_dir=args.output_dir, token=args.token)
