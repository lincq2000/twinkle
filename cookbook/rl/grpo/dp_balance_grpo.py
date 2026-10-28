"""GRPO on GSM8K with DP token balancing (`--balance-batch`).

With padding-free inputs every model DP rank trains on a contiguous slice of each mini-batch, and
the step waits for the rank holding the most tokens. Thinking completions vary a lot in length,
so that slice is often uneven. `plan_dp_balance` reorders the mini-batch before it is dispatched so
every rank gets the same number of samples and a similar token workload. The samples do not
change, so `GRPOLoss` (a per-sequence mean) gives the same update; only the waiting shrinks.

Balancing pays off most when each rank runs one or a few micro-batches per optimizer step, since
the ranks synchronize on every micro-batch. Training metrics gain `dp_balance/*` entries with the
per-rank token counts and the workload imbalance (busiest rank over the mean) before and after.

Linear-attention models such as Qwen3.5 need flash-linear-attention here: their native
GatedDeltaNet does not reset its state between the packed sequences of a padding-free batch.
On NPU that means flash-linear-attention >= 0.5.2, which ships the Ascend Triton backend.

Usage:
    sh dp_balance_grpo.sh                            # accelerate + DDP
    sh dp_balance_grpo.sh --strategy native_fsdp     # FSDP2
    sh dp_balance_grpo.sh --no-balance-batch         # same run without balancing, for comparison
"""
from functools import partial
from typing import Any, Dict, List

from peft import LoraConfig
from transformers import AutoConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.cli import CLI
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.hub import HubOperation
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel, plan_dp_balance
from twinkle.model.dp_balance import estimate_workload
from twinkle.patch import Patch
from twinkle.preprocessor.llm import GSM8KProcessor
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.sampler import vLLMSampler

logger = get_logger()
args = CLI.from_args()

# ========== Configuration ==========
MODEL_ID = args.model.model_id or 'ms://Qwen/Qwen3.5-4B'
TEMPLATE = args.template.template_cls or 'Qwen3_5Template'
DATASET_ID = args.dataset.dataset_id or 'ms://modelscope/gsm8k'
DATASET_SUBSET = 'main' if args.dataset.subset_name == 'default' else args.dataset.subset_name
STRATEGY = args.model.strategy
DEVICE_TYPE = Platform.get_platform().__name__

MODEL_GPUS = args.infra.model_gpus or 2
SAMPLER_GPUS = args.infra.sampler_gpus or 2
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = args.rl.num_generations or 8
MAX_NEW_TOKENS = args.sampling.max_tokens or 2048
LEARNING_RATE = args.optimizer.learning_rate or 1e-5
MAX_STEPS = args.training.max_steps or 200
BATCH_SIZE = args.training.batch_size or 8
MINI_BATCH_SIZE = args.training.mini_batch_size or 16
MICRO_BATCH_SIZE = args.training.micro_batch_size or 8
LORA_RANK = args.lora.lora_r or 16
BALANCE_BATCH = args.training.balance_batch
ATTN_IMPLEMENTATION = args.get('attn_implementation')

SYSTEM_PROMPT = 'You are a helpful math assistant. Reason step by step and put your final answer within \\boxed{}.'


class NPULinearAttentionKernel(Patch):
    """Route Qwen3.5 linear attention through flash-linear-attention on NPU.

    Padding-free inputs need it: the native GatedDeltaNet does not reset its state between the
    packed sequences, and Transformers only reports the fla fast path as available on CUDA-like
    devices. Needs flash-linear-attention >= 0.5.2, whose Ascend backend compiles on NPU.
    """

    def __call__(self, module, *args, **kwargs):
        from twinkle.kernel import KernelChoice, kernelize
        kernelize(module, {'fla': KernelChoice(op='fla', backends=('npu', ))})


def create_gsm8k_dataset():
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta(DATASET_ID, subset_name=DATASET_SUBSET, split='train'))
    # Thinking on and no length penalty: completion lengths differ a lot between prompts.
    dataset.set_template(TEMPLATE, model_id=MODEL_ID, max_length=4096, truncation_strategy='delete',
                         enable_thinking=True)
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type=DEVICE_TYPE),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type=DEVICE_TYPE),
    ]
    if STRATEGY == 'native_fsdp':
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, fsdp_size=MODEL_GPUS)
    else:
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # The attention term of the workload estimate grows with the hidden size.
    config_dir = HubOperation.download_model(MODEL_ID, ignore_model=True)
    hf_config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True)
    hidden_size = (getattr(hf_config, 'text_config', None) or hf_config).hidden_size
    workload_fn = partial(estimate_workload, hidden_size=hidden_size)

    model_kwargs = {'attn_implementation': ATTN_IMPLEMENTATION} if ATTN_IMPLEMENTATION else {}
    model = TransformersModel(
        model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model', strategy=STRATEGY, **model_kwargs)
    if DEVICE_TYPE == 'NPU':
        model.apply_patch(NPULinearAttentionKernel)
    # Language-model projections only (full and linear attention, MLP); 'all-linear' would add the vision tower.
    lora_config = LoraConfig(
        target_modules=[
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'in_proj_qkv', 'in_proj_z',
            'in_proj_a', 'in_proj_b', 'out_proj'
        ],
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05)
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
    model.set_optimizer('AdamW', lr=LEARNING_RATE)
    model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    # Balancing needs padding-free inputs: with fixed padding every sample costs the same.
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template(TEMPLATE, model_id=MODEL_ID, enable_thinking=True)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': max(32, LORA_RANK),
            'enable_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE, model_id=MODEL_ID, enable_thinking=True)
    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler, platform=DEVICE_TYPE)

    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=BATCH_SIZE,
        min_batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )
    accuracy_reward = GSM8KAccuracyReward()
    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, temperature=1.0, top_p=0.95)

    logger.info(get_device_placement())
    logger.info(f'strategy={STRATEGY}, balance_batch={BALANCE_BATCH}, model DP={model_mesh.data_world_size}')
    optim_step = 0
    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break
        metrics.reset()
        expand_prompts = [prompt for prompt in batch for _ in range(NUM_GENERATIONS)]
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()
        sample_responses = sampler.sample(expand_prompts, sampling_params)

        all_inputs: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        completion_lengths: List[int] = []
        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_inputs.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                completion_lengths.append(len(sequence.tokens))

        rewards = accuracy_reward(all_inputs)
        metrics.accumulate(completion_lengths=completion_lengths, rewards={'accuracy': rewards})
        advantages = advantage_fn(rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        balance_metrics: Dict[str, List[float]] = {}
        for mb_start in range(0, len(all_inputs), MINI_BATCH_SIZE):
            mb_inputs = all_inputs[mb_start:mb_start + MINI_BATCH_SIZE]
            mb_old_logps = all_old_logps[mb_start:mb_start + MINI_BATCH_SIZE]
            mb_advantages = advantages[mb_start:mb_start + MINI_BATCH_SIZE]
            if BALANCE_BATCH:
                # Reorder all per-sample lists together so they stay aligned.
                dp_plan = plan_dp_balance(mb_inputs, model_mesh.data_world_size, workload_fn=workload_fn)
                mb_inputs, mb_old_logps, mb_advantages = (
                    dp_plan.apply(values) for values in (mb_inputs, mb_old_logps, mb_advantages))
                for key, value in dp_plan.metrics.items():
                    balance_metrics.setdefault(key, []).append(value)

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1
            if optim_step >= MAX_STEPS:
                break

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict.update({key: sum(values) / len(values) for key, values in balance_metrics.items()})
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('dp-balance-grpo-final')


if __name__ == '__main__':
    main()
