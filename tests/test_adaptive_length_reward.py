import pytest
import torch

from areal.utils.functional import reward_adaptive_length_penalty


def _batch(rewards: list[float], response_lengths: list[int]) -> dict:
    max_len = max(response_lengths)
    loss_mask = torch.zeros((len(rewards), max_len), dtype=torch.int32)
    for i, length in enumerate(response_lengths):
        loss_mask[i, :length] = 1
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "loss_mask": loss_mask,
    }


def test_adaptive_length_penalty_skips_hard_groups():
    data = _batch(
        rewards=[1.0, 0.0, 0.0, 0.0],
        response_lengths=[4000, 6000, 8000, 10000],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        min_correct=1,
        min_solve_rate=0.5,
        target_quantile=0.25,
        min_target_len=2048,
    )

    torch.testing.assert_close(out["adaptive_length_penalties"], torch.zeros(4))
    torch.testing.assert_close(out["adaptive_length_active"], torch.zeros(4))
    torch.testing.assert_close(
        out["adaptive_length_solve_rate"],
        torch.tensor([0.25, 0.25, 0.25, 0.25]),
    )
    torch.testing.assert_close(out["rewards"], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_adaptive_length_penalty_scales_with_group_solve_rate():
    data = _batch(
        rewards=[1.0, 1.0, 0.0, 1.0],
        response_lengths=[4, 8, 10, 6],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        min_correct=2,
        min_solve_rate=0.5,
        max_solve_rate=1.0,
        target_quantile=0.5,
        min_target_len=1,
        max_penalty=1.0,
    )

    torch.testing.assert_close(
        out["adaptive_length_penalties"],
        torch.tensor([0.0, -1.0 / 30.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        out["adaptive_length_active"], torch.tensor([1.0, 1.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        out["adaptive_length_target_len"], torch.tensor([6.0, 6.0, 6.0, 6.0])
    )
    torch.testing.assert_close(
        out["adaptive_length_solve_rate"], torch.tensor([0.75, 0.75, 0.75, 0.75])
    )
    torch.testing.assert_close(
        out["rewards"], torch.tensor([1.0, 29.0 / 30.0, 0.0, 1.0])
    )


def test_adaptive_length_penalty_uses_quantile_not_minimum():
    data = _batch(
        rewards=[1.0, 1.0, 1.0, 1.0],
        response_lengths=[4, 8, 12, 16],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        min_correct=2,
        min_solve_rate=0.5,
        max_solve_rate=1.0,
        target_quantile=0.5,
        min_target_len=1,
        max_penalty=0.15,
    )

    torch.testing.assert_close(
        out["adaptive_length_target_len"], torch.tensor([8.0, 8.0, 8.0, 8.0])
    )
    torch.testing.assert_close(
        out["adaptive_length_penalties"], torch.tensor([0.0, 0.0, -0.1, -0.15])
    )
    torch.testing.assert_close(out["rewards"], torch.tensor([1.0, 1.0, 0.9, 0.85]))


def test_adaptive_length_penalty_alp_mode_uses_absolute_length_cost():
    data = _batch(
        rewards=[1.0, 0.0, 1.0, 0.0],
        response_lengths=[4, 8, 12, 16],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        mode="alp",
        length_normalizer=16,
        max_penalty=None,
    )

    torch.testing.assert_close(
        out["adaptive_length_penalties"],
        torch.tensor([-0.025, -0.05, -0.075, -0.1]),
    )
    torch.testing.assert_close(out["adaptive_length_active"], torch.ones(4))
    torch.testing.assert_close(
        out["adaptive_length_target_len"], torch.full((4,), 16.0)
    )
    torch.testing.assert_close(
        out["adaptive_length_solve_rate"], torch.full((4,), 0.5)
    )
    torch.testing.assert_close(
        out["rewards"], torch.tensor([0.975, -0.05, 0.925, -0.1])
    )


def test_adaptive_length_penalty_alp_mode_keeps_floor_for_unsolved_groups():
    data = _batch(
        rewards=[0.0, 0.0, 0.0, 0.0],
        response_lengths=[4, 8, 12, 16],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        mode="alp",
        length_normalizer=16,
        max_penalty=None,
    )

    torch.testing.assert_close(
        out["adaptive_length_penalties"],
        torch.tensor([-0.0125, -0.025, -0.0375, -0.05]),
    )
    torch.testing.assert_close(out["adaptive_length_solve_rate"], torch.zeros(4))
    torch.testing.assert_close(out["adaptive_length_active"], torch.ones(4))


def test_adaptive_length_penalty_alp_mode_clamps_max_penalty():
    data = _batch(
        rewards=[1.0, 1.0, 1.0, 1.0],
        response_lengths=[4, 8, 12, 16],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=1.0,
        reward_threshold=1.0,
        mode="alp",
        length_normalizer=16,
        max_penalty=0.5,
    )

    torch.testing.assert_close(
        out["adaptive_length_penalties"],
        torch.tensor([-0.25, -0.5, -0.5, -0.5]),
    )


def test_adaptive_length_penalty_alp_mode_uses_max_length_fallback():
    data = _batch(
        rewards=[1.0, 0.0, 0.0, 0.0],
        response_lengths=[4, 8, 12, 16],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        mode="alp",
        length_normalizer=None,
        max_penalty=None,
    )

    torch.testing.assert_close(
        out["adaptive_length_target_len"],
        torch.full((4,), 16.0),
    )
    torch.testing.assert_close(
        out["adaptive_length_penalties"],
        torch.tensor([-0.0125, -0.025, -0.0375, -0.05]),
    )


def test_adaptive_length_penalty_handles_padded_last_group():
    data = _batch(
        rewards=[1.0, 1.0, 0.0],
        response_lengths=[4, 8, 12],
    )

    out = reward_adaptive_length_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        mode="alp",
        length_normalizer=16,
        max_penalty=None,
    )

    torch.testing.assert_close(
        out["adaptive_length_solve_rate"],
        torch.full((3,), 2.0 / 3.0),
    )
    torch.testing.assert_close(
        out["adaptive_length_penalties"],
        torch.tensor([-1.0 / 30.0, -1.0 / 15.0, -0.1]),
    )


def test_adaptive_length_penalty_rejects_unknown_mode():
    data = _batch(rewards=[1.0, 1.0], response_lengths=[4, 8])

    with pytest.raises(ValueError, match="Unknown adaptive length reward mode"):
        reward_adaptive_length_penalty(
            data,
            group_size=2,
            alpha=0.2,
            mode="missing",
        )
