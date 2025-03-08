import hashlib
import os
from typing import Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def get_dataset(
    hub_url: str,
    subset: Optional[str],
    *,
    features: list[str],
    max_seq_length: int,
    tokenizer: AutoTokenizer,
    split: str,
    n_samples: Union[int, Literal["all"]] = "all",
    token=None,
):
    data_stream: Optional[IterableDataset] = None

    if subset is not None:
        data_stream = load_dataset(
            hub_url,
            subset,
            split=split,
            token=token,
            streaming=True,
        )
    else:
        data_stream = load_dataset(
            hub_url,
            split=split,
            token=token,
            streaming=True,
        )

    data_points = []

    for data_point in tqdm(data_stream, desc=f"Loading the {split} data"):
        data_points.append(data_point)
        if n_samples != "all" and len(data_points) >= n_samples:
            break

    def tokenize_text(element):
        encodings = tokenizer(
            element[features[0]],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_length=True,
            return_tensors="pt",
        )

        return encodings

    raw_data = HfDataset.from_list(data_points)
    tokenized_data = raw_data.map(
        tokenize_text,
        batched=True,
        remove_columns=raw_data.column_names,
        desc=f"Tokenizing the {split} data",
    )

    return tokenized_data


def get_cached_dataset(
    hub_url,
    subset,
    features,
    max_seq_length,
    tokenizer,
    split,
    n_samples,
    token=None,
):
    """Get dataset from cache if available, otherwise download and cache it"""
    # Create cache directory
    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Create a unique cache ID based on dataset parameters
    cache_id = f"{hub_url}_{subset or 'none'}_{split}"
    cache_id += f"_{max_seq_length}_{tokenizer.name_or_path.replace('/', '_')}"
    cache_id += f"_{n_samples}_{'_'.join(features)}"
    # Hash the ID to ensure valid filename
    cache_id = hashlib.md5(cache_id.encode()).hexdigest()

    cache_path = os.path.join(cache_dir, cache_id)

    # Check if cached dataset exists
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        return load_from_disk(cache_path)

    # Otherwise download and process
    print(
        f"Downloading dataset: {hub_url} (subset: {subset or 'None'})"
        f" (split: {split}) (samples: {n_samples})"
    )

    dataset = get_dataset(
        hub_url=hub_url,
        subset=subset,
        features=features,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        split=split,
        n_samples=n_samples,
        token=token,
    )

    # Save to disk for future use
    print(f"Saving dataset to {cache_path}")
    dataset.save_to_disk(cache_path)

    return dataset
