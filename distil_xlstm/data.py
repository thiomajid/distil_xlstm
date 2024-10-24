from typing import Optional

from datasets import Dataset as HfDataset
from datasets import load_dataset
from transformers import AutoTokenizer

from distil_xlstm.trainer.trainer_arguments import KDArguments


def get_dataset(args: KDArguments, *, tokenizer: AutoTokenizer, split: str):
    raw_data: Optional[HfDataset] = None

    if args.data_subset is not None:
        raw_data = load_dataset(
            args.dataset_url,
            args.data_subset,
            split=split,
        )
    else:
        raw_data: HfDataset = load_dataset(
            args.dataset_url,
            split=split,
        )

    def tokenize_text(element):
        encodings = tokenizer(
            element["text"],
            truncation=True,
            max_length=args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )

        input_batch = []
        for length, input_ids in zip(encodings["length"], encodings["input_ids"]):
            if length == args.context_length:
                input_batch.append(input_ids)

        return {"input_ids": input_batch, "labels": input_ids}

    tokenized_data = raw_data.map(
        tokenize_text,
        batched=True,
        remove_columns=raw_data.column_names,
    )

    return tokenized_data
