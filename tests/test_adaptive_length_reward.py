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
