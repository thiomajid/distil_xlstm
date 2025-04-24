#! /bin/bash

python3 train_distil_xlstm.py \
    model=train_distil_xlstm \
    ++model.num_blocks_init="same" \
    ++trainer.hub_token="${HUB_TOKEN}" \
    ++trainer.hub_model_id="${HUB_MODEL_ID}" \
    