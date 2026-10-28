# Copyright (c) ModelScope Contributors. All rights reserved.
import random

import pytest
import torch

from twinkle import DeviceMesh
from twinkle.model.dp_balance import DPBalancePlan, balanced_dp_order, effective_length, plan_dp_balance
from twinkle.model.micro_batch import _kk_partition


def _rank_chunks(order, dp_size):
    mesh = DeviceMesh.from_sizes(world_size=dp_size, dp_size=dp_size, device_type='cpu')
    return [order[mesh.get_slice(len(order), rank)] for rank in range(dp_size)]


def _peak(lengths, order, dp_size):
    return max(sum(lengths[index] for index in chunk) for chunk in _rank_chunks(order, dp_size))


def test_equal_size_kk_partition_gives_equal_group_sizes():
    lengths = [random.Random(seed).randint(1, 100) for seed in range(12)]

    groups = _kk_partition(lengths, 4, equal_size=True)

    assert sorted(index for group in groups for index in group) == list(range(12))
    assert [len(group) for group in groups] == [3, 3, 3, 3]
    with pytest.raises(ValueError, match='equal-size'):
        _kk_partition(lengths[:10], 4, equal_size=True)


@pytest.mark.parametrize(('total', 'dp_size'), [(8, 2), (16, 4), (7, 3), (10, 4), (3, 4)])
@pytest.mark.parametrize('seed', range(5))
def test_balanced_order_keeps_get_slice_counts_and_never_raises_the_peak(total, dp_size, seed):
    rng = random.Random(seed)
    lengths = [rng.randint(1, 4096) for _ in range(total)]
    identity = list(range(total))

    order = balanced_dp_order(lengths, dp_size, workload_fn=lambda length: length)

    assert sorted(order) == identity
    assert [len(chunk) for chunk in _rank_chunks(order, dp_size)] == [
        len(chunk) for chunk in _rank_chunks(identity, dp_size)
    ]
    assert all(chunk == sorted(chunk) for chunk in _rank_chunks(order, dp_size))
    assert _peak(lengths, order, dp_size) <= _peak(lengths, identity, dp_size)


def test_balanced_order_spreads_long_sequences_across_ranks():
    lengths = [1000, 1000, 1000, 1000, 10, 10, 10, 10]

    order = balanced_dp_order(lengths, 4, workload_fn=lambda length: length)

    assert [sorted(lengths[index] for index in chunk) for chunk in _rank_chunks(order, 4)] == [[10, 1000]] * 4


def test_balanced_order_keeps_original_order_when_nothing_improves():
    assert balanced_dp_order([5, 1, 9, 3], 1) == [0, 1, 2, 3]
    assert balanced_dp_order([7, 7, 7, 7], 2) == [0, 1, 2, 3]
    assert balanced_dp_order([], 2) == []


def test_group_size_keeps_groups_together_on_one_rank():
    lengths = [900, 800, 700, 600, 30, 20, 10, 5]

    order = balanced_dp_order(lengths, 2, group_size=2, workload_fn=lambda length: length)

    assert order != list(range(8))
    for chunk in _rank_chunks(order, 2):
        assert len(chunk) == 4
        assert [index // 2 for index in chunk[::2]] == [index // 2 for index in chunk[1::2]]
        assert all(second == first + 1 for first, second in zip(chunk[::2], chunk[1::2]))
    with pytest.raises(ValueError, match='divisible by dp_size'):
        balanced_dp_order(lengths, 3, group_size=2)
    with pytest.raises(ValueError, match='multiple of group_size'):
        balanced_dp_order(lengths[:7], 2, group_size=2)


def test_plan_apply_reorders_lists_tuples_and_tensors():
    plan = DPBalancePlan(order=[2, 0, 1])

    assert plan.apply(['a', 'b', 'c']) == ['c', 'a', 'b']
    assert plan.apply((1, 2, 3)) == (3, 1, 2)
    assert plan.apply(torch.tensor([10, 20, 30])).tolist() == [30, 10, 20]
    with pytest.raises(ValueError, match='2 values'):
        plan.apply([1, 2])


def test_plan_dp_balance_counts_real_tokens_and_reports_metrics():
    real_lengths = [8, 6, 2, 1]
    inputs = [{
        'input_ids': list(range(8)),
        'attention_mask': [1] * length + [0] * (8 - length),
    } for length in real_lengths]

    plan = plan_dp_balance(inputs, 2)

    assert [effective_length(model_input) for model_input in inputs] == real_lengths
    assert [real_lengths[index] for index in plan.order] == [8, 1, 6, 2]
    assert plan.metrics['dp_balance/rank_tokens_max_before'] == 14
    assert plan.metrics['dp_balance/rank_tokens_min_before'] == 3
    assert plan.metrics['dp_balance/rank_tokens_max_after'] == 9
    assert plan.metrics['dp_balance/rank_tokens_min_after'] == 8
    assert plan.metrics['dp_balance/rank_tokens_mean'] == 8.5
    assert plan.metrics['dp_balance/workload_imbalance_after'] < plan.metrics['dp_balance/workload_imbalance_before']


def test_effective_length_rejects_unencoded_inputs():
    with pytest.raises(TypeError, match='encoded inputs'):
        effective_length('raw text')
    with pytest.raises(ValueError, match='input_ids'):
        effective_length({'messages': []})
