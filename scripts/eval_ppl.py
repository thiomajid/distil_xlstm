import json
import logging
import math
import time
from pathlib import Path

import hydra
import torch
import torch.utils
import torch.utils.data
from datasets import Dataset as HFDataset
from einops import rearrange
from huggingface_hub import snapshot_download, upload_file
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_xlstm.data import get_dataset
from distil_xlstm.modeling import DistilxLSTMForCausalLM
from distil_xlstm.utils import PerplexityEvaluationConfig, count_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def download_weights(config: PerplexityEvaluationConfig):
    weights_dir = Path(config.local_dir)

    if not weights_dir.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=config.hub_url,
        token=config.hub_token,
        local_dir=weights_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )

    logger.info(f"Model weights downloaded to {weights_dir}")


def load_model(config: PerplexityEvaluationConfig):
    logger.info(f"Loading model from {config.hub_url}")

    model: DistilxLSTMForCausalLM | AutoModelForCausalLM | None = None

    if config.model_type == "hub_model":
        logger.info(f"Downloading model weights for model of type {config.model_type}")
        download_weights(config)

    if config.model_type == "distil_xlstm" or config.model_type == "xlstm":
        model = DistilxLSTMForCausalLM.from_safetensors(
            hf_repo=config.hub_url,
            local_dir=config.local_dir,
            device=config.device,
            token=config.hub_token,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.hub_url,
            token=config.hub_token,
            device_map="auto",
            trust_remote_code=True,
        )

    if config.fp16:
        logger.info("Casting model to half precision (FP16)")
        model = model.to(config.device, dtype=torch.float16)

    logger.info("Downloading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.hub_url, token=config.hub_token)

    return model, tokenizer


def evaluate_perplexity(
    config: PerplexityEvaluationConfig,
    model: torch.nn.Module,
    dataset: HFDataset,
    batch_size: int,
):
    model.eval()
    device = model.device

    total_loss = 0
    total_tokens = 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Convert lists to tensors if needed
            if isinstance(batch["input_ids"], list):
                input_ids = torch.tensor(batch["input_ids"]).to(device)
            else:
                input_ids = batch["input_ids"].to(device)

            # Handle attention mask if available
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                if isinstance(attention_mask, list):
                    attention_mask = torch.tensor(attention_mask).to(device)
                else:
                    attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get logits
            logits: torch.Tensor | None = None
            if config.model_type != "hub_model":
                logits = outputs["logits"]
            else:
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Shift logits and labels for next token prediction
            # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
            shifted_logits = rearrange(
                logits[..., :-1, :].contiguous(), "b s v -> (b s) v"
            )

            # shape: [batch, seq] -> [batch * (seq-1)]
            shifted_labels = rearrange(input_ids[..., 1:].contiguous(), "b s -> (b s)")

            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                input=shifted_logits,
                target=shifted_labels,
            )

            # Apply mask if available to consider only non-padding tokens
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
                loss = loss * shift_mask
                num_tokens = shift_mask.sum().item()
            else:
                num_tokens = loss.numel()

            total_loss += loss.sum().item()
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    return {
        "ce_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens_evaluated": total_tokens,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="eval_config")
def main(cfg: DictConfig):
    config = PerplexityEvaluationConfig(
        **OmegaConf.to_container(cfg["eval"], resolve=True)
    )

    print(config)

    # Load model and tokenizer
    model, tokenizer = load_model(config)
    logger.info(f"Model loaded successfully. Model type: {config.model_type}")
    logger.info(f"Parameter count: {count_parameters(model)}")

    # Load cached dataset or download if not available
    eval_data = get_dataset(
        hub_url=config.dataset_url,
        subset=config.data_subset,
        split=config.data_split,
        features=[config.text_column],
        max_seq_length=config.max_seq_length,
        num_samples=config.samples,
        token=config.hub_token,
        tokenizer=tokenizer,
        trust_remote_code=True,
    )

    # Prepare for PyTorch
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "length"])
    logger.info(f"Dataset ready. Number of samples: {len(eval_data)}")

    # Start timing the evaluation using perf_counter for better precision
    start_time = time.perf_counter()

    # Evaluate perplexity
    results = evaluate_perplexity(
        config=config,
        model=model,
        dataset=eval_data,
        batch_size=config.batch_size,
    )

    # End timing
    end_time = time.perf_counter()
    eval_seconds = end_time - start_time
    eval_minutes = eval_seconds / 60

    # Add timing to results
    model_name = config.hub_url.split("/")[-1]
    dataset_name = config.dataset_url.split("/")[-1]
    results.update(
        {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "evaluation_time_seconds": eval_seconds,
            "evaluation_time_minutes": eval_minutes,
            "tokens_per_second": results["total_tokens_evaluated"] / eval_seconds,
            "model_type": config.model_type,
            "model_path": config.local_dir,
            "dataset": config.dataset_url,
            "dataset_subset": config.data_subset,
            "split": config.data_split,
            "samples_evaluated": len(eval_data),
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "device": config.device,
            "fp16": config.fp16,
        }
    )

    output_file = Path(f"{model_name}_perplexity_eval_on_{dataset_name}.json")

    # Save results to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Upload results to Hugging Face Hub after the evaluation
    if config.hub_token:
        upload_file(
            path_or_fileobj=output_file,
            path_in_repo=output_file.name,
            repo_id=config.hub_url,
            token=config.hub_token,
            commit_message=f"Upload evaluation results for {model_name}",
        )

    logger.info(f"Results saved to {output_file}")
    logger.info(f"CE Loss: {results['ce_loss']:.4f}")
    logger.info(f"Perplexity: {results['perplexity']:.4f}")

    logger.info(
        f"Evaluation time: {int(eval_minutes)}m {int(eval_seconds % 60)}s ({eval_seconds:.2f}s)"
    )

    logger.info(f"Total tokens evaluated: {results['total_tokens_evaluated']}")
    logger.info(f"Throughput: {results['tokens_per_second']:.2f} tokens/second")


if __name__ == "__main__":
    main()
