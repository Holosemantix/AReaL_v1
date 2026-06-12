import functools
from typing import Any, Dict, List, Union, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

logger = logging.getLogger(__name__)


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(
            x_sum_sq,
            op=dist.ReduceOp.SUM,
            group=reduce_group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level geometric mean ratios and average advantages per sequence (GSPO).

    Args:
        log_ratio: Log of probability ratios (logprobs - proximal_logprobs)
        advantages: Per-token advantages
        loss_mask: Boolean mask indicating valid tokens
        cu_seqlens: Cumulative sequence lengths. Required for 1D tensors (packed format).
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            For a single sequence, use cu_seqlens=torch.tensor([0, seq_len]).

    Returns:
        ratio: Sequence-level importance sampling ratios (broadcast to all tokens)
        advantages: Sequence-averaged advantages (broadcast to all tokens)
            Note: We use mean instead of sum to keep gradient magnitude independent
            of sequence length. When multiplied by ratio and summed over tokens,
            this gives the correct total gradient contribution per sequence.
    """
    # Handle both 1D (packed) and 2D (padded) tensor shapes
    if log_ratio.ndim == 1:
        # For 1D tensors (packed format), cu_seqlens is required
        if cu_seqlens is None:
            raise ValueError(
                "cu_seqlens is required for 1D tensors (packed format). "
                "In AReaL, 1D tensors are produced by pack_tensor_dict() and always have cu_seqlens. "
                "For a single sequence, use cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32)."
            )

        # Packed sequences: use cu_seqlens boundaries
        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # Create sequence index for each token: [0,0,0,1,1,2,2,2,2,...]
        sequence_idx = torch.arange(
            batch_size, device=log_ratio.device
        ).repeat_interleave(seq_lengths)

        # Use scatter_add for vectorized summation per sequence (faster than Python loop)
        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(
            batch_size, device=log_ratio.device, dtype=log_ratio.dtype
        ).scatter_add_(0, sequence_idx, masked_log_ratio)

        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(
            batch_size, device=advantages.device, dtype=advantages.dtype
        ).scatter_add_(0, sequence_idx, masked_advantages)

        valid_count_per_seq = (
            torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
            .scatter_add_(0, sequence_idx, loss_mask.int())
            .clamp(min=1)
        )

        # Compute sequence-level means
        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(
            log_ratio.dtype
        )
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(
            advantages.dtype
        )

        # Broadcast sequence-level values back to token-level
        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)

        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        # For 2D tensors (padded sequences)
        # Input shape: [batch_size, seq_len]
        # Compute mean log ratio over sequence length for each sample
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / (
            loss_mask.sum(dim=1).clamp(min=1)
        )
        # Broadcast back to original shape: each sequence gets its own geometric mean ratio
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        # Apply mask
        ratio = torch.where(loss_mask, ratio, 0.0)

        # Average token advantages per sequence
        # This ensures gradient magnitude is independent of sequence length
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        advantages = (advantages.sum(dim=-1, keepdim=True) / seq_lengths).expand_as(
            log_ratio
        )

    return ratio, advantages


def compute_behave_imp_weight(
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    behave_imp_weight_mode: str,
    behave_imp_weight_cap: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute behavioural importance weight for decoupled loss correction.

    Args:
        proximal_logprobs: Recomputed log probabilities from reference model
        old_logprobs: Log probabilities from inference engine
        loss_mask: Boolean mask indicating valid tokens
        cu_seqlens: Cumulative sequence lengths for packed sequences
        behave_imp_weight_mode: Mode for importance weight correction
            - 'token_truncate': clamp token ratio to [0, cap]
            - 'token_mask': set token ratio to 0 where ratio > cap
            - 'sequence_truncate': clamp sequence ratio to [0, cap]
            - 'sequence_mask': set sequence ratio to 0 where ratio > cap
            - 'disabled': skip importance weight correction
        behave_imp_weight_cap: Cap value for importance weights

    Returns:
        Tuple of (behave_imp_weight, behave_approx_kl, behave_mask)
    """
    if behave_imp_weight_mode == "disabled":
        raise ValueError(
            "compute_behave_imp_weight should not be called with mode='disabled'. "
            "The caller should guard this call with 'if behave_imp_weight_mode != \"disabled\"'."
        )

    is_sequence_level = "sequence" in behave_imp_weight_mode
    behave_approx_kl = proximal_logprobs - old_logprobs
    behave_imp_weight_log_ratio = behave_approx_kl

    if is_sequence_level:
        # Compute sequence-level geometric mean importance weights
        dummy_advantages = torch.zeros_like(behave_imp_weight_log_ratio)
        behave_imp_weight_seq, _ = _compute_sequence_level_ratio_and_advantages(
            behave_imp_weight_log_ratio,
            dummy_advantages,
            loss_mask,
            cu_seqlens,
        )
        behave_imp_weight = behave_imp_weight_seq
    else:
        # Token-level importance weights (default)
        behave_imp_weight = behave_imp_weight_log_ratio.exp()

    # Apply cap (truncate or mask) based on mode
    if behave_imp_weight_cap is not None:
        if "truncate" in behave_imp_weight_mode:
            behave_imp_weight = behave_imp_weight.clamp(
                min=0.0, max=behave_imp_weight_cap
            )
        else:  # mask
            behave_imp_weight = torch.where(
                behave_imp_weight > behave_imp_weight_cap, 0.0, behave_imp_weight
            )

    # Apply loss_mask
    behave_imp_weight = torch.where(loss_mask, behave_imp_weight, 0.0)
    behave_mask = (behave_imp_weight > 0).logical_and(loss_mask)
    behave_approx_kl = torch.where(behave_mask, behave_approx_kl, 0.0)

    return behave_imp_weight, behave_approx_kl, behave_mask


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    behave_imp_weight_cap: float | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
    behave_imp_weight_mode: str = "token_mask",
) -> tuple[torch.Tensor, dict]:
    """
    When decoupled loss is disabled:
    1. if recompute logp, both old_logprobs and proximal_logprobs are recomputed logp;
    2. if no recomputation, both old_logp and proximal_logprobs are produced by the inference backend.

    When decoupled loss is enabled, proximal_logprobs is the recomputed logp,
    old_logprobs is produced by the inference engine.

    Args:
        importance_sampling_level: Level at which to compute importance sampling ratios.
            - 'token': Per-token ratios
            - 'sequence': Sequence-level geometric mean of per-token ratios (GSPO)
        cu_seqlens: Cumulative sequence lengths for packed sequences (1D tensors).
            Required when inputs are 1D and importance_sampling_level='sequence'.
            Shape: [batch_size + 1], where cu_seqlens[i] marks the start of sequence i.
            Not needed for 2D padded inputs (sequences identified by batch dimension).
        behave_imp_weight_mode: Mode for importance weight correction (mask or truncate).
            - 'token_truncate': clamp token ratio to [0, cap]
            - 'token_mask': set token ratio to 0 where ratio > cap
            - 'sequence_truncate': clamp sequence ratio to [0, cap]
            - 'sequence_mask': set sequence ratio to 0 where ratio > cap
            - 'disabled': skip importance weight correction
    """
    loss_mask_count = loss_mask.count_nonzero() or 1

    if importance_sampling_level == "sequence":
        # GSPO: Compute sequence-level geometric mean of probability ratios
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        # Standard PPO: per-token ratio
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - eps_clip,
        1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher),
    )

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    # Compute behavioural importance weight only when not disabled
    # When disabled, pg_loss remains unchanged (no behavioural correction applied)
    if behave_imp_weight_mode != "disabled":
        behave_imp_weight, behave_approx_kl, behave_mask = compute_behave_imp_weight(
            proximal_logprobs=proximal_logprobs,
            old_logprobs=old_logprobs,
            loss_mask=loss_mask,
            cu_seqlens=cu_seqlens,
            behave_imp_weight_mode=behave_imp_weight_mode,
            behave_imp_weight_cap=behave_imp_weight_cap,
        )
        pg_loss = pg_loss * behave_imp_weight

    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
    if proximal_logprobs is not None and behave_imp_weight_mode != "disabled":
        stat.update(
            behave_approx_kl=behave_approx_kl.detach(),
            behave_imp_weight=behave_imp_weight.detach(),
            behave_mask=behave_mask,
        )
    return pg_loss, stat


def sapo_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    loss_mask: torch.Tensor,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """SAPO (Soft Adaptive Policy Optimization) loss with asymmetric sigmoid gates.

    SAPO replaces PPO clipping with soft sigmoid gates, providing smooth gradients.
    Note: SAPO requires use_decoupled_loss=False.

    Args:
        logprobs: Current policy log probabilities
        old_logprobs: Old policy log probabilities
        advantages: Advantage values
        tau_pos: Temperature for positive advantages (higher = sharper gate)
        tau_neg: Temperature for negative advantages (higher = sharper gate)
        loss_mask: Mask for valid tokens
        importance_sampling_level: "token" or "sequence" level importance sampling
        cu_seqlens: Cumulative sequence lengths for sequence-level IS

    Returns:
        Tuple of (loss, statistics dict compatible with PPO)
    """
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("SAPO temperatures (tau_pos, tau_neg) must be positive.")
    loss_mask_count = loss_mask.count_nonzero() or 1
    advantages = advantages.detach()
    log_ratio = logprobs - old_logprobs

    if importance_sampling_level == "sequence":
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
    elif importance_sampling_level == "token":
        ratio = torch.exp(log_ratio)
    else:
        raise ValueError(
            f"Invalid importance_sampling_level: {importance_sampling_level}. "
            "Must be 'token' or 'sequence'."
        )

    # SAPO: Asymmetric sigmoid gates with 4/τ gradient normalization
    gate_pos = torch.sigmoid(tau_pos * (ratio - 1.0))
    gate_neg = torch.sigmoid(tau_neg * (ratio - 1.0))
    scale_pos = 4.0 / tau_pos
    scale_neg = 4.0 / tau_neg
    scaled_gate_pos = gate_pos * scale_pos
    scaled_gate_neg = gate_neg * scale_neg

    # Select gate based on advantage sign
    is_positive = advantages > 0
    soft_gate = torch.where(is_positive, scaled_gate_pos, scaled_gate_neg)

    # Compute loss
    pg_loss = -soft_gate * advantages
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count

    # Return stat dict compatible with PPO (fake clip_mask for logging compatibility)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),  # SAPO doesn't clip
        dual_clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),
        # SAPO-specific stats (scaled gates for consistency)
        sapo_soft_gate=soft_gate.detach(),
        sapo_scaled_gate_pos=scaled_gate_pos.detach(),
        sapo_scaled_gate_neg=scaled_gate_neg.detach(),
    )

    return pg_loss, stat


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float):
    diff = torch.abs(x - y)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))


def _mse_loss(x: torch.Tensor, y: torch.Tensor):
    return 0.5 * (x - y) ** 2


def ppo_critic_loss_fn(
    value: torch.FloatTensor,
    old_value: torch.FloatTensor,
    target_value: torch.FloatTensor,
    value_eps_clip: float,
    loss_mask: torch.Tensor | None = None,
    loss_fn_type: str = "mse",
) -> tuple[torch.Tensor, dict]:
    """Compute PPO critic loss function given padded batch inputs.

    There is no shape requirements for the inputs, but they must have the same shape.
    Either [bs, max_seqlen] for batch padded inputs or [tot_seqlen] for padded inputs.

    Args:
        value (torch.FloatTensor): Values. The position of the final token is not included.
            (The whole generated sequence is not a state.)
        old_value (torch.FloatTensor): Old values.
        target_value (torch.FloatTensor): Returns computed by GAE.
        value_eps_clip (float): Clip ratio.
        loss_mask (Optional[torch.Tensor], optional): Mask for loss computation.
            1 if valid else 0. Defaults to None.
        loss_fn_type (str, optional): Type of loss function. Defaults to 'mse'.

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """
    assert value.dtype == torch.float32
    assert old_value.dtype == torch.float32
    assert target_value.dtype == torch.float32

    if loss_fn_type == "huber":
        loss_fn = functools.partial(_huber_loss, delta=10.0)
    elif loss_fn_type == "mse":
        loss_fn = _mse_loss
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    if target_value.is_inference():
        target_value = target_value.clone()  # clone a inference tensor

    value_loss_original = loss_fn(value, target_value)

    value_clipped = old_value + (value - old_value).clamp(
        -value_eps_clip, value_eps_clip
    )

    value_loss_clipped = loss_fn(value_clipped, target_value)

    value_loss = torch.max(value_loss_original, value_loss_clipped)

    with torch.no_grad():
        clip_mask = value_loss_clipped.detach() > value_loss_original.detach()
        if loss_mask is not None:
            clip_mask.logical_and_(loss_mask)

        stat = dict(clip_mask=clip_mask, loss=value_loss.detach())

    if loss_mask is not None:
        value_loss = (
            torch.where(loss_mask, value_loss, 0).sum() / loss_mask.count_nonzero()
        )
    else:
        value_loss = value_loss.mean()

    return value_loss, stat


# code modified from VERL: https://github.com/volcengine/verl/blob/main/verl/workers/reward_manager/dapo.py
def reward_overlong_penalty(
    data: dict[str, Any],
    overlong_tokens: int,
    overlong_penalty_factor: float,
    max_response_length: int,
) -> dict[str, Any]:
    reward_score = data["rewards"]
    if "raw_task_rewards" not in data:
        data["raw_task_rewards"] = reward_score.detach().clone()

    response_lengths = data["loss_mask"].sum(dim=-1).to(dtype=reward_score.dtype)
    expected_len = max_response_length - overlong_tokens
    exceed_len = response_lengths - expected_len
    overlong_penalties = torch.minimum(
        -exceed_len / overlong_tokens * overlong_penalty_factor,
        torch.zeros_like(reward_score),
    )

    data["overlong_penalties"] = overlong_penalties
    data["rewards"] = reward_score + overlong_penalties
    return data


def reward_shortest_correct_penalty(
    data: dict[str, Any],
    group_size: int,
    alpha: float,
    reward_threshold: float = 1.0,
    min_correct: int = 2,
    normalize_by_shortest: bool = True,
    max_penalty: float | None = None,
    min_shortest_len: int = 1,
) -> dict[str, Any]:
    """Penalize correct samples that are longer than the shortest correct peer."""
    reward_score = data["rewards"]
    penalties = torch.zeros_like(reward_score)

    if group_size <= 1 or alpha <= 0 or reward_score.numel() == 0:
        data["shortest_correct_penalties"] = penalties
        data["shortest_correct_active"] = torch.zeros_like(reward_score)
        data["shortest_correct_target_len"] = torch.zeros_like(reward_score)
        return data

    raw_rewards = data.get("raw_task_rewards", reward_score).to(dtype=reward_score.dtype)
    response_lengths = data["loss_mask"].sum(dim=-1).to(dtype=reward_score.dtype)

    bs = reward_score.shape[0]
    padded_bs = ((bs + group_size - 1) // group_size) * group_size
    pad = padded_bs - bs

    valid = torch.ones_like(reward_score, dtype=torch.bool)
    if pad:
        raw_rewards = F.pad(raw_rewards, (0, pad), value=float("-inf"))
        response_lengths = F.pad(response_lengths, (0, pad), value=0.0)
        valid = F.pad(valid, (0, pad), value=False)

    group_rewards = raw_rewards.view(-1, group_size)
    group_lengths = response_lengths.view(-1, group_size)
    group_valid = valid.view(-1, group_size)

    correct = (group_rewards >= reward_threshold) & group_valid
    correct_count = correct.sum(dim=-1, keepdim=True)
    active_group = correct_count >= min_correct

    inf_lengths = torch.full_like(group_lengths, float("inf"))
    shortest_correct_len = torch.where(correct, group_lengths, inf_lengths).amin(
        dim=-1, keepdim=True
    )
    shortest_correct_len = shortest_correct_len.clamp(min=float(min_shortest_len))

    excess = (group_lengths - shortest_correct_len).clamp(min=0.0)
    denominator = shortest_correct_len if normalize_by_shortest else 1.0
    group_penalties = -alpha * excess / denominator
    group_penalties = torch.where(
        correct & active_group,
        group_penalties,
        torch.zeros_like(group_penalties),
    )
    if max_penalty is not None and max_penalty > 0:
        group_penalties = group_penalties.clamp(min=-max_penalty)

    penalties = group_penalties.reshape(-1)[:bs]
    active = (correct & active_group).to(dtype=reward_score.dtype).reshape(-1)[:bs]
    targets = torch.where(
        active_group.expand_as(group_lengths),
        shortest_correct_len.expand_as(group_lengths),
        torch.zeros_like(group_lengths),
    ).reshape(-1)[:bs]

    data["shortest_correct_penalties"] = penalties
    data["shortest_correct_active"] = active
    data["shortest_correct_target_len"] = targets
    data["rewards"] = reward_score + penalties
    return data


def reward_adaptive_length_penalty(
    data: dict[str, Any],
    group_size: int,
    alpha: float,
    reward_threshold: float = 1.0,
    min_correct: int = 1,
    min_solve_rate: float = 0.5,
    max_solve_rate: float = 1.0,
    target_quantile: float = 0.25,
    min_target_len: int = 1,
    normalize_by_target: bool = True,
    max_penalty: float | None = None,
    correct_only: bool = True,
) -> dict[str, Any]:
    """Difficulty-aware length penalty scaled by group solve rate.

    Easy groups receive stronger length pressure. Hard groups below
    ``min_solve_rate`` receive no length penalty, preserving reasoning budget.
    The target length is a correct-sample length quantile instead of the minimum
    correct length, avoiding the collapse mode seen with shortest-correct rewards.
    """
    reward_score = data["rewards"]
    penalties = torch.zeros_like(reward_score)
    zeros = torch.zeros_like(reward_score)

    if group_size <= 1 or alpha <= 0 or reward_score.numel() == 0:
        data["adaptive_length_penalties"] = penalties
        data["adaptive_length_active"] = zeros
        data["adaptive_length_target_len"] = zeros
        data["adaptive_length_solve_rate"] = zeros
        return data

    target_quantile = min(max(target_quantile, 0.0), 1.0)
    raw_rewards = data.get("raw_task_rewards", reward_score).to(dtype=reward_score.dtype)
    response_lengths = data["loss_mask"].sum(dim=-1).to(dtype=reward_score.dtype)

    bs = reward_score.shape[0]
    padded_bs = ((bs + group_size - 1) // group_size) * group_size
    pad = padded_bs - bs

    valid = torch.ones_like(reward_score, dtype=torch.bool)
    if pad:
        raw_rewards = F.pad(raw_rewards, (0, pad), value=float("-inf"))
        response_lengths = F.pad(response_lengths, (0, pad), value=0.0)
        valid = F.pad(valid, (0, pad), value=False)

    group_rewards = raw_rewards.view(-1, group_size)
    group_lengths = response_lengths.view(-1, group_size)
    group_valid = valid.view(-1, group_size)

    correct = (group_rewards >= reward_threshold) & group_valid
    correct_count = correct.sum(dim=-1, keepdim=True)
    valid_count = group_valid.sum(dim=-1, keepdim=True).clamp(min=1)
    solve_rate = correct_count.to(dtype=reward_score.dtype) / valid_count.to(
        dtype=reward_score.dtype
    )

    active_group = (correct_count >= min_correct) & (solve_rate >= min_solve_rate)
    if max_solve_rate > min_solve_rate:
        solve_scale = (
            (solve_rate - min_solve_rate) / (max_solve_rate - min_solve_rate)
        ).clamp(min=0.0, max=1.0)
    else:
        solve_scale = (solve_rate >= min_solve_rate).to(dtype=reward_score.dtype)

    inf_lengths = torch.full_like(group_lengths, float("inf"))
    correct_lengths = torch.where(correct, group_lengths, inf_lengths)
    sorted_correct_lengths = correct_lengths.sort(dim=-1).values
    target_index = torch.floor(
        (correct_count.to(dtype=reward_score.dtype) - 1.0).clamp(min=0.0)
        * target_quantile
    ).to(dtype=torch.long)
    target_len = sorted_correct_lengths.gather(dim=-1, index=target_index)
    target_len = target_len.clamp(min=float(min_target_len))

    sample_mask = correct if correct_only else group_valid
    active = sample_mask & active_group
    excess = (group_lengths - target_len).clamp(min=0.0)
    denominator = target_len if normalize_by_target else 1.0
    group_penalties = -alpha * solve_scale * excess / denominator
    group_penalties = torch.where(
        active, group_penalties, torch.zeros_like(group_penalties)
    )
    if max_penalty is not None and max_penalty > 0:
        group_penalties = group_penalties.clamp(min=-max_penalty)

    penalties = group_penalties.reshape(-1)[:bs]
    active_values = active.to(dtype=reward_score.dtype).reshape(-1)[:bs]
    target_values = torch.where(
        active_group.expand_as(group_lengths),
        target_len.expand_as(group_lengths),
        torch.zeros_like(group_lengths),
    ).reshape(-1)[:bs]
    solve_rate_values = torch.where(
        group_valid,
        solve_rate.expand_as(group_lengths),
        torch.zeros_like(group_lengths),
    ).reshape(-1)[:bs]

    data["adaptive_length_penalties"] = penalties
    data["adaptive_length_active"] = active_values
    data["adaptive_length_target_len"] = target_values
    data["adaptive_length_solve_rate"] = solve_rate_values
    data["rewards"] = reward_score + penalties
    return data


# =============================================================================
# New Functions for Decomposed IG / PRM Reward Calculation
# =============================================================================
def build_ig_probe_batches(
        data: Dict[str, Any],
        mini_batch_size: int,
        sep_token_id: Union[List[int], List[List[int]]],
        pad_token_id: int,
        step_separation_mode: str = "separator",
        uncertainty_threshold: float = -1.5,
        min_step_tokens: int = 5,
        max_step_tokens: int = 128  # [新增] 限制最大截断区间
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    outcome_rewards = data["rewards"]

    behavior_logprobs = data.get("logprobs", None)

    ig_gt_ids_batch = data["ig_gt_ids"]
    ig_bridge_len_batch = data["ig_bridge_len"]
    ig_gt_len_batch = data["ig_gt_len"]

    batch_size = input_ids.shape[0]
    device = input_ids.device

    all_probes_info = []

    for i in range(batch_size):
        nonzero_indices = torch.nonzero(loss_mask[i])
        if len(nonzero_indices) > 0:
            prompt_len = nonzero_indices[0].item()
        else:
            prompt_len = len(input_ids[i])

        curr_input_ids = input_ids[i]
        curr_ig_gt_ids = ig_gt_ids_batch[i]
        curr_bridge_len = ig_bridge_len_batch[i]
        curr_gt_len = ig_gt_len_batch[i]
        curr_outcome = outcome_rewards[i]

        curr_logprobs = None
        if step_separation_mode == "uncertainty" and behavior_logprobs is not None:
            curr_logprobs = behavior_logprobs[i]

        probes_seqs, metadata, step_end_indices = _prepare_single_sample_probes(
            input_ids=curr_input_ids,
            prompt_len=prompt_len,
            ig_gt_ids=curr_ig_gt_ids,
            ig_bridge_len=curr_bridge_len,
            ig_gt_len=curr_gt_len,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
            step_separation_mode=step_separation_mode,
            logprobs=curr_logprobs,
            uncertainty_threshold=uncertainty_threshold,
            min_step_tokens=min_step_tokens,
            max_step_tokens=max_step_tokens  # [新增] 传入提取逻辑
        )

        all_probes_info.append({
            "sample_idx": i,
            "probes_seqs": probes_seqs,
            "metadata": metadata,
            "step_end_indices": step_end_indices,
            "outcome_reward": curr_outcome,
            "num_probes": len(probes_seqs),
            "gt_len": curr_gt_len.item(),
            "prompt_len": prompt_len
        })

    if batch_size > 0:
        if all_probes_info[0]["num_probes"] == 0:
            logger.warning(
                f"[IG-Debug] No steps detected for sample 0! "
                f"Mode: {step_separation_mode}. "
                f"Input IDs sample (last 20): {input_ids[0, -20:].tolist()}"
            )
        elif logger.isEnabledFor(logging.INFO):
            logger.info(
                f"[IG-Debug] Detected {all_probes_info[0]['num_probes']} steps for sample 0 (Mode: {step_separation_mode}).")

    flat_probes_seqs = []
    flat_metadata = []
    current_probe_idx = 0

    for info in all_probes_info:
        seqs = info["probes_seqs"]
        info["start_probe_idx"] = current_probe_idx
        flat_probes_seqs.extend(seqs)
        flat_metadata.extend(info["metadata"])
        current_probe_idx += len(seqs)

    probe_batches = []
    total_probes = len(flat_probes_seqs)

    if total_probes > 0:
        for i in range(0, total_probes, mini_batch_size):
            chunk_seqs = flat_probes_seqs[i: i + mini_batch_size]

            max_len = max([seq.size(1) for seq in chunk_seqs])
            padded_chunk_inputs = []
            for seq in chunk_seqs:
                curr_len = seq.size(1)
                pad_len = max_len - curr_len
                if pad_len > 0:
                    padded_seq = F.pad(seq, (0, pad_len), value=pad_token_id)
                else:
                    padded_seq = seq
                padded_chunk_inputs.append(padded_seq)

            batch_input_ids = torch.cat(padded_chunk_inputs, dim=0)
            batch_attention_mask = (batch_input_ids != pad_token_id).long()
            batch_position_ids = torch.arange(max_len, device=device).unsqueeze(0).expand(len(chunk_seqs), -1)

            probe_data = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "position_ids": batch_position_ids
            }
            probe_batches.append(probe_data)

    full_metadata = {
        "all_probes_info": all_probes_info,
        "flat_metadata": flat_metadata
    }

    return probe_batches, full_metadata


def assign_ig_rewards(
        data: Dict[str, Any],
        logprobs_list: List[torch.Tensor],
        metadata_dict: Dict[str, Any],
        beta: float = 0.5,
        use_peak_selection: bool = False,
        use_watermark_selection: bool = False,  # [新增参数] 最大单调递增子序列过滤
        reward_mode: str = "prob_diff"
) -> Dict[str, Any]:
    input_ids = data["input_ids"]
    device = input_ids.device
    all_probes_info = metadata_dict["all_probes_info"]
    flat_metadata = metadata_dict["flat_metadata"]

    # --- 核心改造：独立初始化结构分、逻辑分与显式Mask ---
    if "token_level_rewards" not in data:
        data["token_level_rewards"] = torch.zeros_like(input_ids, dtype=torch.float32, device=device)
    if "step_boundary_mask" not in data:
        data["step_boundary_mask"] = torch.ones_like(input_ids, dtype=torch.float32, device=device)
    if "token_level_len_penalties" not in data:
        data["token_level_len_penalties"] = torch.zeros_like(input_ids, dtype=torch.float32, device=device)
    # 新增：显式的步骤掩码。之前使用 (raw_step_rewards != 0) 过滤，
    # 但由于 Hindsight 约束可能会把无效步骤的奖励强行置为 0.0，这会导致步骤被漏算，
    # 破坏了 RMS Norm 的基数计算，现在我们使用显式的 1.0/0.0 mask 记录有效步骤。
    if "step_reward_mask" not in data:
        data["step_reward_mask"] = torch.zeros_like(input_ids, dtype=torch.float32, device=device)

    all_log_prob_sums = []
    current_flat_idx = 0

    for batch_logprobs in logprobs_list:
        batch_size = batch_logprobs.shape[0]

        for k in range(batch_size):
            if current_flat_idx >= len(flat_metadata):
                break

            start, end = flat_metadata[current_flat_idx]
            seq_len = batch_logprobs.shape[1]
            valid_end = min(end - 1, seq_len)
            valid_start = max(0, start - 1)

            if valid_start < valid_end:
                lp_sum = batch_logprobs[k, valid_start:valid_end].sum()
            else:
                lp_sum = torch.tensor(0.0, device=device)

            all_log_prob_sums.append(lp_sum)
            current_flat_idx += 1

    for info in all_probes_info:
        start_idx = info["start_probe_idx"]
        num = info["num_probes"]
        step_end_indices = info["step_end_indices"]
        outcome_reward = info["outcome_reward"]
        sample_idx = info["sample_idx"]
        gt_len = info.get("gt_len", 1)
        prompt_len = info.get("prompt_len", 0)

        if num == 0: continue
        if start_idx + num > len(all_log_prob_sums):
            logger.error(f"[IG-Error] Missing logprobs for sample {sample_idx}")
            continue

        sample_log_probs = all_log_prob_sums[start_idx: start_idx + num]

        # 提取真实步长 (Token 数量)
        step_lengths = []
        last_idx = prompt_len - 1
        for idx in step_end_indices:
            step_lengths.append(idx - last_idx)
            last_idx = idx

        # 获取解耦后的 逻辑奖励 和 长度惩罚
        calculated_rewards, calculated_penalties = _compute_rewards_logic(
            log_probs=sample_log_probs,
            outcome_reward=outcome_reward,
            beta=beta,
            use_peak_selection=use_peak_selection,
            use_watermark_selection=use_watermark_selection,  # 传入底层
            reward_mode=reward_mode,
            gt_len=gt_len,
            step_lengths=step_lengths
        )

        for r, p, idx in zip(calculated_rewards, calculated_penalties, step_end_indices):
            if idx > 0 and idx <= input_ids.shape[1]:
                # 纯粹的逻辑跃升奖励
                data["token_level_rewards"][sample_idx, idx - 1] += r
                # 纯粹的结构长度惩罚
                data["token_level_len_penalties"][sample_idx, idx - 1] += p
                # 显式打上有效步骤标记（即使 r == 0.0）
                data["step_reward_mask"][sample_idx, idx - 1] = 1.0

                if idx < input_ids.shape[1]:
                    data["step_boundary_mask"][sample_idx, idx] = 0.0

    return data


def _prepare_single_sample_probes(
        input_ids,
        prompt_len,
        ig_gt_ids,
        ig_bridge_len,
        ig_gt_len,
        sep_token_id: Union[List[int], List[List[int]]],
        pad_token_id: int,
        step_separation_mode: str = "separator",
        logprobs: torch.Tensor | None = None,
        uncertainty_threshold: float = -1.5,
        min_step_tokens: int = 5,
        max_step_tokens: int = 128  # [新增] 防止 Qwen BPE 粘连导致的无节制膨胀
):
    if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
    if ig_gt_ids.dim() == 1: ig_gt_ids = ig_gt_ids.unsqueeze(0)

    q_tensor = input_ids[:, :prompt_len]
    reasoning_ids = input_ids[:, prompt_len:]
    reasoning_list = reasoning_ids[0].tolist()

    step_rel_indices = []

    if step_separation_mode == "separator":
        separator_candidates = []
        if len(sep_token_id) > 0:
            if isinstance(sep_token_id[0], list):
                separator_candidates = sep_token_id
            else:
                separator_candidates = [sep_token_id]

        if len(separator_candidates) > 0:
            i = 0
            last_cut_idx = -1  # 记录上一次切分的位置
            while i < len(reasoning_list):
                matched = False
                for cand in separator_candidates:
                    cand_len = len(cand)
                    if i + cand_len <= len(reasoning_list):
                        if reasoning_list[i: i + cand_len] == cand:
                            # 确保切分出来的 step 长度 >= min_step_tokens
                            end_idx = i + cand_len - 1
                            if end_idx - last_cut_idx >= min_step_tokens:
                                step_rel_indices.append(end_idx)
                                last_cut_idx = end_idx
                            i += cand_len
                            matched = True
                            break
                if not matched:
                    # [新增] 兜底保护机制：如果距离上一次切分已经超过 max_step_tokens，强制切断
                    # 避免因为 Qwen 的 Tokenizer 将 \n 和其他符号合并，导致整个序列无法正常切分
                    if (i - last_cut_idx) >= max_step_tokens:
                        step_rel_indices.append(i)
                        last_cut_idx = i
                    i += 1

    elif step_separation_mode == "uncertainty":
        if logprobs is not None:
            if logprobs.dim() == 1:
                reasoning_logprobs = logprobs[prompt_len:]
            else:
                reasoning_logprobs = logprobs.squeeze()[prompt_len:]

            last_idx = -1
            for i in range(len(reasoning_logprobs)):
                val = reasoning_logprobs[i].item()
                if val < uncertainty_threshold:
                    potential_end_idx = i - 1
                    if potential_end_idx - last_idx >= min_step_tokens:
                        if potential_end_idx >= 0:
                            step_rel_indices.append(potential_end_idx)
                            last_idx = potential_end_idx
                # [新增] 同样在不确定度模式下加入最大步长保护
                elif (i - last_idx) >= max_step_tokens:
                    step_rel_indices.append(i)
                    last_idx = i
        else:
            logger.warning("[IG] Uncertainty mode selected but logprobs not provided.")

    if reasoning_ids.shape[1] > 0:
        final_idx = reasoning_ids.shape[1] - 1
        if not step_rel_indices or step_rel_indices[-1] != final_idx:
            step_rel_indices.append(final_idx)

    suffix_tensor = ig_gt_ids
    q_len = q_tensor.shape[1]

    suffix_gt_start_idx = ig_bridge_len.item()
    suffix_gt_end_idx = suffix_gt_start_idx + ig_gt_len.item()

    probe_seqs = []
    metadata = []
    step_abs_indices = []

    init_seq = torch.cat([q_tensor, suffix_tensor], dim=1)
    target_start_0 = q_len + suffix_gt_start_idx
    target_end_0 = q_len + suffix_gt_end_idx
    probe_seqs.append(init_seq)
    metadata.append((target_start_0, target_end_0))

    for rel_end_idx in step_rel_indices:
        current_cot = reasoning_ids[:, :rel_end_idx + 1]
        probe_seq = torch.cat([q_tensor, current_cot, suffix_tensor], dim=1)
        prefix_len = q_len + current_cot.shape[1]
        target_start = prefix_len + suffix_gt_start_idx
        target_end = prefix_len + suffix_gt_end_idx
        probe_seqs.append(probe_seq)
        metadata.append((target_start, target_end))
        step_abs_indices.append(prompt_len + rel_end_idx)

    return probe_seqs, metadata, step_abs_indices


def _compute_rewards_logic(
        log_probs,
        outcome_reward,
        beta,
        use_peak_selection=False,
        use_watermark_selection=False,
        max_abs_step_reward=5.0,
        reward_mode="prob_diff",
        gt_len=1,
        step_lengths=None
):
    T = len(log_probs) - 1
    is_correct = outcome_reward > 0
    final_rewards = []
    final_penalties = []
    step_probs = []
    device = log_probs[0].device if isinstance(log_probs[0], torch.Tensor) else torch.device("cpu")
    log_probs = [torch.as_tensor(lp, dtype=torch.float32, device=device) for lp in log_probs]

    norm_len = max(1, gt_len)
    # 提取 P_0 作为初始水位
    p_0_tensor = torch.exp(log_probs[0] / norm_len)
    initial_watermark = p_0_tensor.item() if isinstance(p_0_tensor, torch.Tensor) else p_0_tensor

    for t in range(1, T + 1):
        raw_r_t = torch.tensor(0.0).to(device) if device else 0.0

        p_t_absolute = torch.exp(log_probs[t] / norm_len)
        step_probs.append(p_t_absolute)

        if t == T:
            raw_r_t = torch.tensor(0.0).to(device) if device else 0.0
        else:
            if reward_mode == "prob_diff":
                p_t = torch.exp(log_probs[t] / norm_len)
                p_prev = torch.exp(log_probs[t - 1] / norm_len)
                raw_r_t = p_t - p_prev

            elif reward_mode == "prob_diff_hindsight":
                p_t = torch.exp(log_probs[t] / norm_len)
                p_prev = torch.exp(log_probs[t - 1] / norm_len)
                raw_r_t = p_t - p_prev

                if not is_correct:
                    raw_r_t = torch.clamp(raw_r_t, max=0.0)

            elif reward_mode == "prob":
                if is_correct:
                    raw_r_t = torch.exp(log_probs[t] / norm_len)
            else:
                if is_correct:
                    raw_r_t = log_probs[t] - log_probs[t - 1]

        clipped_r_t = torch.clamp(raw_r_t, -max_abs_step_reward, max_abs_step_reward)

        # [修改] 暂时不用长度惩罚，强置为 0.0
        len_penalty = 0.0
        final_rewards.append(clipped_r_t)
        final_penalties.append(torch.tensor(len_penalty).to(device) if device else len_penalty)

    def get_val(idx):
        r = final_rewards[idx]
        return r.item() if isinstance(r, torch.Tensor) else r

    def get_prob(idx):
        p = step_probs[idx]
        return p.item() if isinstance(p, torch.Tensor) else p

    # 1. 最高水位线 (Highest Watermark) 过滤与结算逻辑
    # [修改] 不再区分是否正确，所有轨迹均参与水位线计算以提供密集的正面引导
    if use_watermark_selection and len(final_rewards) > 0:
        new_rewards = []
        new_penalties = []
        running_max = initial_watermark

        for i in range(len(final_rewards)):
            # 强制保留最后一步（输出答案）无过程奖励的规则
            if i == len(final_rewards) - 1:
                new_rewards.append(torch.tensor(0.0).to(device) if device else 0.0)
                new_penalties.append(final_penalties[i])
                continue

            # (此处的 if not is_correct 阻断逻辑已被彻底移除)

            p_t = get_prob(i)
            # 核心机制：只有当前概率突破历史最高水位线，才发放严密的增量奖励
            if p_t > running_max:
                raw_r = p_t - running_max
                clipped_r = torch.clamp(
                    torch.tensor(raw_r, dtype=torch.float32, device=device) if device else raw_r,
                    -max_abs_step_reward, max_abs_step_reward
                )
                new_rewards.append(clipped_r)
                new_penalties.append(final_penalties[i])
                # 刷新水位线
                running_max = p_t
            else:
                # 未突破水位线，过滤掉（逻辑奖励置 0，长度惩罚也置 0 以免引入噪声，水位保持不变）
                new_rewards.append(torch.tensor(0.0, dtype=torch.float32, device=device) if device else 0.0)
                new_penalties.append(torch.tensor(0.0, dtype=torch.float32, device=device) if device else 0.0)

        final_rewards = new_rewards
        final_penalties = new_penalties

    # 2. 原有的局部峰值 (Peak) 过滤逻辑（与Watermark互斥）
    elif use_peak_selection and len(final_rewards) > 0:
        new_rewards = []
        new_penalties = []
        n = len(final_rewards)

        for i in range(n):
            val = get_val(i)
            is_peak = True

            if i > 0 and val <= get_val(i - 1):
                is_peak = False
            if i < n - 1 and val <= get_val(i + 1):
                is_peak = False

            if is_peak:
                new_rewards.append(final_rewards[i])
                new_penalties.append(final_penalties[i])
            else:
                new_rewards.append(torch.tensor(0.0).to(device) if device else 0.0)
                new_penalties.append(torch.tensor(0.0).to(device) if device else 0.0)

        final_rewards = new_rewards
        final_penalties = new_penalties

    return final_rewards, final_penalties
