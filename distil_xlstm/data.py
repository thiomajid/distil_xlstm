from typing import Optional

from datasets import Dataset as HfDataset
from datasets import load_dataset
from transformers import AutoTokenizer

from distil_xlstm.trainer.trainer_arguments import KDArguments


def get_dataset(args: KDArguments, *, tokenizer: AutoTokenizer, split: str):
    raw_data: Optional[HfDataset] = None

    n_samples = range(args.train_samples)

    if args.data_subset is not None:
        raw_data = load_dataset(
            args.dataset_url,
            args.data_subset,
            split=split,
        ).select(n_samples)
    else:
        raw_data = load_dataset(
            args.dataset_url,
            split=split,
        ).select(n_samples)

    def tokenize_text(element):
        encodings = tokenizer(
            element["text"],
            truncation=True,
            max_length=args.context_length,
            padding="max_length",
            return_length=True,
            return_tensors="pt",
        )

        return encodings

    tokenized_data = raw_data.map(
        tokenize_text,
        batched=True,
        remove_columns=raw_data.column_names,
        desc="Tokenizing the dataset",
    )

    return tokenized_data
