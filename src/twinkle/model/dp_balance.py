# Copyright (c) ModelScope Contributors. All rights reserved.
"""Data-parallel token load balancing.

``slice_dp`` dispatch hands every DP rank one contiguous ``get_slice`` chunk, and the
step waits for the slowest rank. ``plan_dp_balance`` computes a permutation of the
batch so those chunks carry similar compute, without changing how many samples each
rank receives.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .micro_batch import _kk_partition, sequence_length


def estimate_workload(length: int, hidden_size: int = 4096) -> int:
    """Relative forward cost of one dense-transformer sequence.

    Projections and MLP cost about ``12·h²·s`` FLOPs and attention ``2·h·s²``; dividing
    by ``2h`` keeps the ratio and gives ``6·h·s + s²``. ``hidden_size=4096`` reproduces
    verl's ``24576·s + s²``.
    """
    return 6 * hidden_size * length + length * length


def effective_length(model_input: Any) -> int:
    """Number of real tokens in one encoded sample."""
    if not isinstance(model_input, Mapping):
        raise TypeError(f'DP balancing needs encoded inputs (mappings), got {type(model_input).__name__}')
    mask = model_input.get('attention_mask')
    if mask is not None:
        return int(mask.sum()) if hasattr(mask, 'sum') else int(sum(mask))
    if 'input_ids' not in model_input:
        raise ValueError('DP balancing needs encoded inputs with input_ids or attention_mask')
    return sequence_length(model_input)


def _rank_sizes(total: int, dp_size: int) -> list[int]:
    # Mirrors DeviceMesh.get_slice: the first ``total % dp_size`` ranks get one extra item.
    base, remainder = divmod(total, dp_size)
    return [base + int(rank < remainder) for rank in range(dp_size)]


def _per_rank_sums(values: Sequence[int], order: Sequence[int], dp_size: int) -> list[int]:
    sums = []
    start = 0
    for size in _rank_sizes(len(order), dp_size):
        sums.append(sum(values[index] for index in order[start:start + size]))
        start += size
    return sums


def _imbalance(per_rank: Sequence[int]) -> float:
    # max/mean estimates how much slower the step is than a perfectly even split.
    mean = sum(per_rank) / len(per_rank)
    return max(per_rank) / mean if mean else 1.0


def balanced_dp_order(
    lengths: Sequence[int],
    dp_size: int,
    *,
    group_size: int = 1,
    workload_fn: Callable[[int], int] = estimate_workload,
) -> list[int]:
    """Return a permutation whose contiguous ``get_slice`` chunks have balanced workload.

    Every rank keeps the sample count ``get_slice`` would give it, so per-rank loss
    weighting is unchanged. ``group_size`` consecutive samples stay on the same rank
    (for example chosen/rejected pairs); within a rank, samples keep their original
    relative order. The original order is returned when balancing would not lower the
    heaviest rank's workload.
    """
    total = len(lengths)
    if dp_size <= 0:
        raise ValueError(f'dp_size must be positive, got {dp_size}')
    if group_size <= 0 or total % group_size:
        raise ValueError(f'batch size {total} must be a positive multiple of group_size={group_size}')
    if dp_size == 1 or total == 0:
        return list(range(total))
    num_groups = total // group_size
    if group_size > 1 and num_groups % dp_size:
        raise ValueError(f'keeping groups of {group_size} together needs the group count ({num_groups}) '
                         f'to be divisible by dp_size={dp_size}')

    sample_workloads = [workload_fn(length) for length in lengths]
    if min(sample_workloads) < 0:
        raise ValueError(f'workload_fn must return non-negative values, got {min(sample_workloads)}')
    group_workloads = [
        sum(sample_workloads[group * group_size:(group + 1) * group_size]) for group in range(num_groups)
    ]

    # Zero-cost placeholders round the batch up to equal-size partitions. They take the
    # lowest indices so they all land in the first KK seed, hence at most one per rank.
    num_placeholders = -num_groups % dp_size
    partitions = _kk_partition([0] * num_placeholders + group_workloads, dp_size, equal_size=True)
    groups_per_rank = [
        sorted(index - num_placeholders for index in partition if index >= num_placeholders) for partition in partitions
    ]
    # Ranks without a placeholder hold one more group; get_slice puts the larger chunks first.
    groups_per_rank.sort(key=len, reverse=True)

    order = []
    for rank_groups in groups_per_rank:
        for group in rank_groups:
            order.extend(range(group * group_size, (group + 1) * group_size))
    identity = list(range(total))
    # KK is a heuristic, so keep the original order unless the heaviest rank actually gets lighter.
    balanced_peak = max(_per_rank_sums(sample_workloads, order, dp_size))
    original_peak = max(_per_rank_sums(sample_workloads, identity, dp_size))
    return order if balanced_peak < original_peak else identity


@dataclass(frozen=True)
class DPBalancePlan:
    """A batch permutation for ``slice_dp`` dispatch plus before/after balance metrics."""

    order: list[int]
    metrics: dict[str, float] = field(default_factory=dict)

    def apply(self, values: Any) -> Any:
        """Reorder one per-sample field (list, tuple, tensor or array) with the plan."""
        if len(values) != len(self.order):
            raise ValueError(f'cannot reorder {len(values)} values with a plan for {len(self.order)} samples')
        if isinstance(values, list):
            return [values[index] for index in self.order]
        if isinstance(values, tuple):
            return tuple(values[index] for index in self.order)
        if hasattr(values, 'shape'):
            return values[self.order]
        raise TypeError(f'cannot reorder a field of type {type(values).__name__}')


def plan_dp_balance(
    inputs: Sequence[Any],
    dp_size: int,
    *,
    group_size: int = 1,
    workload_fn: Callable[[int], int] = estimate_workload,
    metric_prefix: str = 'dp_balance',
) -> DPBalancePlan:
    """Plan a DP-balanced order for encoded ``inputs`` that will be dispatched with ``slice_dp``.

    Apply the plan to ``inputs`` and every aligned per-sample field (old logps,
    advantages, ...) before calling the model. Results returned by the model come back
    in the permuted order.
    """
    lengths = [effective_length(model_input) for model_input in inputs]
    order = balanced_dp_order(lengths, dp_size, group_size=group_size, workload_fn=workload_fn)
    if not order:
        return DPBalancePlan(order=order)

    identity = list(range(len(order)))
    workloads = [workload_fn(length) for length in lengths]
    tokens_before = _per_rank_sums(lengths, identity, dp_size)
    tokens_after = _per_rank_sums(lengths, order, dp_size)
    workload_before = _per_rank_sums(workloads, identity, dp_size)
    workload_after = _per_rank_sums(workloads, order, dp_size)
    return DPBalancePlan(
        order=order,
        metrics={
            f'{metric_prefix}/rank_tokens_max_before': max(tokens_before),
            f'{metric_prefix}/rank_tokens_min_before': min(tokens_before),
            f'{metric_prefix}/rank_tokens_max_after': max(tokens_after),
            f'{metric_prefix}/rank_tokens_min_after': min(tokens_after),
            f'{metric_prefix}/rank_tokens_mean': sum(lengths) / dp_size,
            f'{metric_prefix}/workload_imbalance_before': _imbalance(workload_before),
            f'{metric_prefix}/workload_imbalance_after': _imbalance(workload_after),
        },
    )
