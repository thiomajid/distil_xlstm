#! /bin/bash

echo "Initializing Gemma 2 training script"

echo "The current working directory is: $(pwd)" && ls
echo "The current date and time is: $(date)"

SCRIPT_FILE="$(pwd)/train_hf.py"
if [ ! -f "$SCRIPT_FILE" ]; then
	echo "Error: $SCRIPT_FILE not found!"
	exit 1
fi

echo "The script file exists: $SCRIPT_FILE"
echo "The script file is executable: $(test -x "$SCRIPT_FILE" && echo 'yes' || echo 'no')"


python3 train_hf.py model=gemma2 \
		+hub_model_id="google/gemma-2-2b" \
		+max_seq_length=256 \
		+attn_implementation="eager" \
		++model.num_hidden_layers=8 \
		++model.hidden_size=960 \
		++model.vocab_size=49152 \
		++trainer.hub_token="${HUB_TOKEN}" \
		++trainer.hub_model_id="${HUB_MODEL_ID}" \
		++trainer.seed=42 \
		++trainer.optim="adamw_torch_fused" \
		++trainer.lr_scheduler_type="cosine" \
		++trainer.learning_rate=5e-5 \
		++trainer.weight_decay=0.01 \
		++trainer.warmup_ratio=0.05 \
		++trainer.fp16=true \
		++trainer.num_train_epochs=1 \
		++trainer.gradient_accumulation_steps=5 \
		++trainer.per_device_train_batch_size=4 \
		++trainer.per_device_eval_batch_size=4 \
		++trainer.logging_steps=200 \
		++trainer.save_steps=200 \
		++trainer.train_dataset_url="HuggingFaceFW/fineweb-edu-llama3-annotations" \
		++trainer.train_split="train" \
		++trainer.train_samples=20000 \
		++trainer.eval_dataset_url="HuggingFaceFW/fineweb-edu" \
		++trainer.eval_split="train" \
		++trainer.eval_subset="default" \
		++trainer.eval_samples=2000 \
		++trainer.features=["text"] \
		++trainer.use_dataset_cache=true \
		++trainer.dataset_cache_dir="./.hf_data_cache"
