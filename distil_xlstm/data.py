from typing import Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import load_dataset
from transformers import AutoTokenizer

from distil_xlstm.trainer.trainer_arguments import KDArguments


def get_dataset(
    args: KDArguments,
    *,
    max_seq_length: int,
    tokenizer: AutoTokenizer,
    split: str,
    n_samples: Union[int, Literal["all"]] = "all",
):
    raw_data: Optional[HfDataset] = None

    if n_samples != "all":
        split = f"{split}[:{n_samples}]"

    if args.data_subset is not None:
        raw_data = load_dataset(
            args.dataset_url,
            args.data_subset,
            split=split,
        )
    else:
        raw_data = load_dataset(
            args.dataset_url,
            split=split,
        )

    def tokenize_text(element):
        encodings = tokenizer(
            element["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_length=True,
            return_tensors="pt",
        )

        return encodings

    tokenized_data = raw_data.map(
        tokenize_text,
        batched=True,
        remove_columns=raw_data.column_names,
        desc=f"Tokenizing the {split} data",
    )

    return tokenized_data
