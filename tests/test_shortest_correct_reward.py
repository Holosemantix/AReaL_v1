import torch

from areal.utils.functional import (
    reward_overlong_penalty,
    reward_shortest_correct_penalty,
)


def _batch(rewards: list[float], response_lengths: list[int]) -> dict:
    max_len = max(response_lengths)
    loss_mask = torch.zeros((len(rewards), max_len), dtype=torch.int32)
    for i, length in enumerate(response_lengths):
        loss_mask[i, :length] = 1
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "loss_mask": loss_mask,
    }


def test_shortest_correct_penalty_uses_group_shortest_correct_length():
    data = _batch(
        rewards=[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        response_lengths=[4, 8, 2, 6, 10, 9, 8, 7],
    )

    out = reward_shortest_correct_penalty(
        data,
        group_size=4,
        alpha=0.2,
        reward_threshold=1.0,
        min_correct=2,
        max_penalty=1.0,
    )

    torch.testing.assert_close(
        out["shortest_correct_penalties"],
        torch.tensor([0.0, -0.2, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        out["rewards"],
        torch.tensor([1.0, 0.8, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0]),
    )


def test_shortest_correct_penalty_ignores_partial_rewards_by_threshold():
    data = _batch(
        rewards=[1.0, 0.5, 1.0, 0.0],
        response_lengths=[10, 4, 6, 8],
    )

    out = reward_shortest_correct_penalty(
        data,
        group_size=4,
        alpha=0.5,
        reward_threshold=1.0,
        min_correct=2,
        max_penalty=1.0,
    )

    torch.testing.assert_close(
        out["shortest_correct_penalties"],
        torch.tensor([-1.0 / 3.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        out["rewards"],
        torch.tensor([2.0 / 3.0, 0.5, 1.0, 0.0]),
    )


def test_shortest_correct_penalty_adds_to_overlong_penalty():
    data = _batch(
        rewards=[1.0, 1.0, 0.0, 1.0],
        response_lengths=[10, 6, 9, 8],
    )
    data["raw_task_rewards"] = data["rewards"].detach().clone()

    data = reward_overlong_penalty(
        data,
        overlong_tokens=2,
        overlong_penalty_factor=1.0,
        max_response_length=10,
    )
    out = reward_shortest_correct_penalty(
        data,
        group_size=4,
        alpha=0.5,
        reward_threshold=1.0,
        min_correct=2,
        max_penalty=1.0,
    )

    torch.testing.assert_close(
        out["overlong_penalties"],
        torch.tensor([-1.0, 0.0, -0.5, 0.0]),
    )
    torch.testing.assert_close(
        out["shortest_correct_penalties"],
        torch.tensor([-1.0 / 3.0, 0.0, 0.0, -1.0 / 6.0]),
    )
    torch.testing.assert_close(
        out["rewards"],
        torch.tensor([-1.0 / 3.0, 1.0, -0.5, 5.0 / 6.0]),
    )
