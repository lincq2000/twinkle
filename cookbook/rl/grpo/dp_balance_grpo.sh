#!/bin/sh
set -eu

# GRPO on GSM8K with DP token balancing across the model DP ranks.
# Thinking completions vary a lot in length; --balance-batch evens out the tokens per rank.
# Each rank runs one micro-batch per optimizer step (mini-batch 16 = 2 ranks x 8), where balancing
# pays off most. Override at invocation, e.g.:
#   sh dp_balance_grpo.sh --strategy native_fsdp      # FSDP2 instead of accelerate + DDP
#   sh dp_balance_grpo.sh --no-balance-batch          # baseline without balancing

python dp_balance_grpo.py \
    --model-id ms://Qwen/Qwen3.5-4B \
    --template-cls Qwen3_5Template \
    --model-gpus 2 \
    --sampler-gpus 2 \
    --num-generations 8 \
    --max-tokens 2048 \
    --batch-size 8 \
    --mini-batch-size 16 \
    --micro-batch-size 8 \
    --max-steps 200 \
    --lr 1e-5 \
    --lora-r 16 \
    --balance-batch \
    "$@"
