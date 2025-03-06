import argparse
import hashlib
import json
import math
import os
import time

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_xlstm import DistilxLSTM
from distil_xlstm.data import get_dataset
from distil_xlstm.utils import count_parameters

#!/usr/bin/env python


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model perplexity on a dataset"
    )

    parser.add_argument(
        "--model-type",
        choices=["distil_xlstm", "huggingface"],
        required=True,
        help="Type of model to use (distil_xlstm from safetensors or huggingface)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="For distil_xlstm: path to safetensors file; For huggingface: model ID",
    )

    parser.add_argument(
        "--hf-repo",
        type=str,
        help="HuggingFace repository for DistilxLSTM configuration (required for distil_xlstm)",
    )

    parser.add_argument(
        "--dataset-url",
        type=str,
        required=True,
        help="URL or name of the dataset on HuggingFace",
    )

    parser.add_argument(
        "--dataset-subset",
        type=str,
        default=None,
        help="Subset of the dataset (optional)",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to use for evaluation",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=["text"],
        help="Feature column(s) to use from dataset",
    )

    parser.add_argument(
        "--samples",
        type=str,
        default="all",
        help="Number of samples to evaluate (integer or 'all')",
    )

    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (cuda/cpu/tpu)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="eval_results.json",
        help="Path to save evaluation results JSON",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision for evaluation",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token for private models (optional)",
    )

    args = parser.parse_args()

    if args.samples != "all":
        args.samples = int(args.samples)

    if args.model_type == "distil_xlstm" and not args.hf_repo:
        parser.error("--hf-repo is required when --model-type is distil_xlstm")

    return args


def load_model(args):
    print(f"Loading model from {args.model_path}")

    if args.model_type == "distil_xlstm":
        model = DistilxLSTM.from_safetensors(
            hf_repo=args.hf_repo,
            filename=args.model_path,
            device=args.device,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.hf_repo, token=args.hf_token)
    else:  # huggingface
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            token=args.hf_token,
            torch_dtype=torch.float16 if args.fp16 else None,
        )

        model = model.to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=args.hf_token)

        # Add padding token if needed
        # if tokenizer.pad_token is None:
        #     if tokenizer.eos_token:
        #         tokenizer.pad_token = tokenizer.eos_token
        #     else:
        #         tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        #         model.resize_token_embeddings(len(tokenizer))

    if args.fp16 and args.model_type == "distil_xlstm":
        model = model.half()

    return model, tokenizer


def evaluate_perplexity(model, dataset, batch_size, device):
    model.eval()
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
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Calculate loss
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(shift_labels.size())

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


# Add function to get cached or fresh dataset
def get_cached_dataset(args, tokenizer):
    # Create cache directory
    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique cache ID based on dataset parameters
    cache_id = f"{args.dataset_url}_{args.dataset_subset or 'none'}_{args.split}"
    cache_id += f"_{args.max_seq_length}_{tokenizer.name_or_path.replace('/', '_')}"
    cache_id += f"_{args.samples}_{'_'.join(args.features)}"
    # Hash the ID to ensure valid filename
    cache_id = hashlib.md5(cache_id.encode()).hexdigest()

    cache_path = os.path.join(cache_dir, cache_id)

    # Check if cached dataset exists
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        return load_from_disk(cache_path)

    # Otherwise download and process
    print(
        f"Downloading dataset: {args.dataset_url} (subset: {args.dataset_subset or 'None'})"
        f" (split: {args.split}) (samples: {args.samples})"
    )

    eval_data = get_dataset(
        hub_url=args.dataset_url,
        subset=args.dataset_subset,
        features=args.features,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        split=args.split,
        n_samples=args.samples,
    )

    # Save to disk for future use
    print(f"Saving dataset to {cache_path}")
    eval_data.save_to_disk(cache_path)

    return eval_data


def main():
    args = parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model(args)
    print(f"Model loaded successfully. Model type: {args.model_type}")
    print(f"Parameter count: {count_parameters(model)}")

    # Load cached dataset or download if not available
    eval_data = get_cached_dataset(args, tokenizer)

    # Prepare for PyTorch
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "length"])
    print(f"Dataset ready. Number of samples: {len(eval_data)}")

    # Start timing the evaluation using perf_counter for better precision
    start_time = time.perf_counter()

    # Evaluate perplexity
    results = evaluate_perplexity(
        model=model, dataset=eval_data, batch_size=args.batch_size, device=args.device
    )

    # End timing
    end_time = time.perf_counter()
    eval_seconds = end_time - start_time
    eval_minutes = eval_seconds / 60

    # Add timing to results
    results.update(
        {
            "evaluation_time_seconds": eval_seconds,
            "evaluation_time_minutes": eval_minutes,
            "tokens_per_second": results["total_tokens_evaluated"] / eval_seconds,
            "model_type": args.model_type,
            "model_path": args.model_path,
            "dataset": args.dataset_url,
            "dataset_subset": args.dataset_subset,
            "split": args.split,
            "samples_evaluated": len(eval_data),
            "batch_size": args.batch_size,
            "max_seq_length": args.max_seq_length,
            "device": args.device,
            "fp16": args.fp16,
        }
    )

    if args.model_type == "distil_xlstm":
        results["hf_repo"] = args.hf_repo

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_file}")
    print(f"CE Loss: {results['ce_loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.4f}")
    print(
        f"Evaluation time: {int(eval_minutes)}m {int(eval_seconds % 60)}s ({eval_seconds:.2f}s)"
    )
    print(f"Throughput: {results['tokens_per_second']:.2f} tokens/second")


if __name__ == "__main__":
    main()
