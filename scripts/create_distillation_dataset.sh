#!/bin/bash

# Script to run the distillation dataset creation with different configurations
# Usage: ./create_distillation_dataset.sh [model_name] [dataset_name]

# Default values
MODEL_NAME=${1:-"qwen"}
DATASET_NAME=${2:-"c4"}
OUTPUT_NAME=${3:-"distil_xlstm_${MODEL_NAME}_${DATASET_NAME}"}

# Check for HUB_TOKEN
if [ -z "$HUB_TOKEN" ]; then
  echo "Error: HUB_TOKEN environment variable is not set"
  echo "Please set it with: export HUB_TOKEN=your_huggingface_token"
  exit 1
fi

echo "Creating distillation dataset with model: $MODEL_NAME and dataset: $DATASET_NAME"

python3 create_distillation_dataset.py \
  teacher_name="$(python -c "import yaml; print(yaml.safe_load(open('configs/distillation/models.yaml'))['$MODEL_NAME']['teacher_name'])")" \
  dataset_url="$(python -c "import yaml; print(yaml.safe_load(open('configs/distillation/datasets.yaml'))['$DATASET_NAME']['dataset_url'])")" \
  data_subset="$(python -c "import yaml; print(yaml.safe_load(open('configs/distillation/datasets.yaml'))['$DATASET_NAME']['data_subset'])")" \
  data_split="$(python -c "import yaml; print(yaml.safe_load(open('configs/distillation/datasets.yaml'))['$DATASET_NAME']['data_split'])")" \
  text_column="$(python -c "import yaml; print(yaml.safe_load(open('configs/distillation/datasets.yaml'))['$DATASET_NAME']['text_column'])")" \
  output_dataset_name="thiomajid/$OUTPUT_NAME" \
  local_dir="./distillation_dataset/$OUTPUT_NAME"
