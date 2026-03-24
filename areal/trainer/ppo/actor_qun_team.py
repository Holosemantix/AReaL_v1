import functools
import sys
import traceback
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed as dist

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.api.engine_api import TrainEngine
from areal.infra import TrainController
from areal.utils import logging, stats_tracker
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.data import (
    KLEstimator,
    Normalization,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional import (
    ppo_actor_loss_fn,
    reward_overlong_penalty,
    sapo_loss_fn,
)
from areal.utils.perf_tracer import trace_perf

logger = logging.getLogger("PPOActor")


class PPOActor:
    def __init__(self, config: PPOActorConfig, engine: TrainEngine):
        self.config = config
        self.engine = engine

        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero

        self.temperature = config.temperature

        self.m2_threshold = config.m2_threshold

        # Log critical GSPO/GRPO configuration for reproducibility
        self._log_configuration()

    def _log_configuration(self):
        """Log PPO configuration including how proximal policy is computed."""
        config = self.config

        logger.info("=" * 70)
        logger.info("PPOActor Configuration")
        logger.info("=" * 70)

        # Log PPO mode and proximal policy computation
        if not config.use_decoupled_loss:
            logger.info("Mode: Standard PPO (on-policy)")
            if config.recompute_logprob:
                logger.info("  old_logp (π_old): RECOMPUTED from current policy")
            else:
                logger.info(
                    "  old_logp (π_old): FROM INFERENCE (cached during rollout)"
                )
        else:
            logger.info("Mode: Decoupled PPO (off-policy)")
            logger.info("  log_p_behave (π_behave): FROM INFERENCE (behavior policy)")

            # Log proximal policy computation method
            method_descriptions = {
                PROX_LOGP_METHOD_RECOMPUTE: "RECOMPUTED via forward pass (standard decoupled PPO)",
                PROX_LOGP_METHOD_LOGLINEAR: "LOG-LINEAR APPROXIMATION (no forward pass)",
                PROX_LOGP_METHOD_METRICS: "RECOMPUTED + APPROXIMATION METRICS (for evaluation)",
            }
            desc = method_descriptions.get(
                config.prox_logp_method, f"UNKNOWN ({config.prox_logp_method})"
            )
            logger.info(f"  Proximal policy (π_prox): {desc}")

            logger.info("  log_p_theta (π_θ): TRAINING FORWARD PASS (current policy)")

            if config.behave_imp_weight_cap:
                logger.info(
                    f"  Importance weight cap: {config.behave_imp_weight_cap:.1f} "
                    "(filters out tokens with extreme weights)"
                )

        # Log other critical config
        logger.info("=" * 70)
        logger.info("Training Parameters:")
        logger.info(
            f"  importance_sampling_level: {getattr(config, 'importance_sampling_level', 'token')}"
        )
        logger.info(
            f"  adv_norm: {config.adv_norm if config.adv_norm else 'DISABLED (None)'}"
        )
        logger.info(
            f"  reward_norm: {config.reward_norm if config.reward_norm else 'DISABLED (None)'}"
        )
        logger.info(f"  eps_clip: {config.eps_clip}")
        logger.info("=" * 70)

    def _safe_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """安全寻找 Actor 的隔离通信组，并加入严苛的尺寸校验（兼容单卡 Debug 模式）"""
        if not dist.is_initialized():
            return tensor

        sync_group = None
        global_ws = dist.get_world_size()

        # 尝试动态获取预期的 DP size (排除 Controller)
        expected_dp_size = -1
        if hasattr(self.engine, "dp_world_size"):
            expected_dp_size = getattr(self.engine, "dp_world_size")
        elif hasattr(self.engine, "data_parallel_size"):
            expected_dp_size = getattr(self.engine, "data_parallel_size")

        def is_valid_group(g):
            try:
                if g is None: return False
                ws = dist.get_world_size(g)

                # 1. 如果明确指定了 expected_dp_size，严格遵循
                if expected_dp_size > 0:
                    return ws == expected_dp_size

                # 2. 如果全局只有 1 张卡（单卡 Debug 模式），必须允许 size=1 的组
                if global_ws == 1:
                    return ws == 1

                # 3. 在多卡环境下，过滤掉 size=1 的无用组（如单卡 TP/PP 组）
                if ws <= 1:
                    return False

                # 4. 兜底策略：允许大小不超过全局 WORLD_SIZE 的组
                return ws <= global_ws
            except Exception:
                return False

        # 1. 优先尝试从 engine 查找（最符合 FSDP 等底层逻辑）
        for attr in ["dp_group", "data_parallel_group", "dp_process_group"]:
            if hasattr(self.engine, attr) and is_valid_group(getattr(self.engine, attr)):
                sync_group = getattr(self.engine, attr)
                break

        # 2. 尝试从自身查找
        if sync_group is None:
            for attr in ["dp_group", "data_parallel_group", "process_group"]:
                if hasattr(self, attr) and is_valid_group(getattr(self, attr)):
                    sync_group = getattr(self, attr)
                    break

        # 3. 尝试使用 Megatron 的 MPU
        if sync_group is None:
            try:
                from megatron.core import mpu
                if mpu.is_initialized() and is_valid_group(mpu.get_data_parallel_group()):
                    sync_group = mpu.get_data_parallel_group()
            except ImportError:
                pass

        if sync_group is None:
            rank = dist.get_rank()
            err = f"[Critical Error] 未能找到任何有效的 Actor DP 通信组！Global WS: {global_ws}, Expected DP: {expected_dp_size}。主动阻断！"
            with open(f"/tmp/areal_crash_rank_{rank}.log", "w") as f:
                f.write(err)
            raise RuntimeError(err)

        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=sync_group)
        return tensor

    def _brutal_crash_reporter(self, method_name: str, exception: Exception, extra_info: str = ""):
        """对抗 Ray 日志黑洞的终极报错拦截器"""
        rank = dist.get_rank() if dist.is_initialized() else 0
        err_msg = f"💥 [CRITICAL CRASH] Rank {rank} in {method_name} 💥\n"
        err_msg += f"Exception: {str(exception)}\n"
        err_msg += f"Context: {extra_info}\n"
        err_msg += f"Traceback:\n{traceback.format_exc()}\n"

        try:
            with open(f"/tmp/areal_crash_rank_{rank}.log", "w") as f:
                f.write(err_msg)
        except:
            pass

        print(err_msg, file=sys.stderr, flush=True)
        logger.error(err_msg)
        raise exception

    @trace_perf("ppo_actor.compute_logp", category="compute")
    @torch.no_grad()
    def compute_logp(self, data: dict[str, Any]) -> torch.Tensor:
        self.engine.eval()

        try:
            res = self.engine.forward(
                input_=data,
                aggregate_fn=lambda xs: torch.cat(xs, dim=-1),
            )
            return res
        except Exception as e:
            info = f"Keys: {list(data.keys())} | input_ids shape: {data.get('input_ids', torch.tensor([])).shape}"
            self._brutal_crash_reporter("compute_logp", e, info)

    @trace_perf("ppo_actor.prepare_ig_probes", category="compute")
    def prepare_ig_probes(self, data: dict[str, Any]) -> tuple[list[dict], Any, int]:
        """
        优雅的 O(1) 通信与极致省内存版本 (Memory-Optimized Prepare Probes)
        通过预分配内存+切片赋值+即时垃圾回收，完全避免了因 F.pad 和 torch.cat 生成多重副本次生对象导致 OOM Killer 的问题。
        """
        try:
            ig_params = self.config.ig_reward_params
            from areal.utils.functional.functional import build_ig_probe_batches

            probe_batches, metadata = build_ig_probe_batches(
                data=data,
                mini_batch_size=ig_params.mini_batch_size,
                sep_token_id=ig_params.sep_token_id,
                pad_token_id=ig_params.pad_token_id,
                step_separation_mode=getattr(ig_params, "step_separation_mode", "separator"),
                uncertainty_threshold=getattr(ig_params, "uncertainty_threshold", -1.5),
                min_step_tokens=getattr(ig_params, "min_step_tokens", 5),
                max_step_tokens=getattr(ig_params, "max_step_tokens", 128),  # [新增] 传入最大区间兜底
            )

            local_count = len(probe_batches)
            device = data["input_ids"].device

            # 1. 统计并收集所有 batch 的形状要求 (省去内部循环调用的数十次通信)
            if dist.is_initialized():
                count_tensor = torch.tensor([local_count], dtype=torch.long, device=device)
                count_tensor = self._safe_all_reduce(count_tensor)
                global_max_count = count_tensor.item()
            else:
                global_max_count = local_count

            if global_max_count == 0:
                metadata["valid_batch_sizes"] = []
                return [], metadata, 0

            # 提取有效的 batch size 用于过程奖励阻断 (必须在改变形状前截取)
            valid_batch_sizes = [b["input_ids"].size(0) for b in probe_batches]
            metadata["valid_batch_sizes"] = valid_batch_sizes

            # 2. O(1) 全局收集目标尺寸: shape_tensor[i, 0] 是 BatchSize, shape_tensor[i, 1] 是 SeqLen
            shape_tensor = torch.zeros((global_max_count, 2), dtype=torch.long, device=device)
            for i in range(local_count):
                shape_tensor[i, 0] = probe_batches[i]["input_ids"].size(0)
                shape_tensor[i, 1] = probe_batches[i]["input_ids"].size(1)

            if dist.is_initialized():
                shape_tensor = self._safe_all_reduce(shape_tensor)

            min_mbs = self.config.ig_reward_params.mini_batch_size

            pad_id = ig_params.pad_token_id if ig_params.pad_token_id is not None else 0
            aligned_batches = []

            # 3. 基于形状预分配内存，避免中间张量碎片（Memory Peak Reduction）
            for i in range(global_max_count):
                target_bs = shape_tensor[i, 0].item()
                target_seq = shape_tensor[i, 1].item()

                # 保底机制：极端情况下全网都是 0 的空载
                target_bs = max(min_mbs, target_bs)
                target_seq = max(2, target_seq)

                # 极致预分配：一次性切出一整块最终内存 (严格保证 contiguous)
                padded_input_ids = torch.full((target_bs, target_seq), pad_id, dtype=torch.long, device=device)
                padded_attention_mask = torch.zeros((target_bs, target_seq), dtype=torch.long, device=device)
                padded_position_ids = torch.arange(target_seq, device=device).unsqueeze(0).expand(target_bs,
                                                                                                  -1).contiguous()

                # 将本地有效数据填入预分配的块内 (无需 cat/pad 产生二次开销)
                if i < local_count:
                    local_bs = probe_batches[i]["input_ids"].size(0)
                    local_seq = probe_batches[i]["input_ids"].size(1)

                    padded_input_ids[:local_bs, :local_seq] = probe_batches[i]["input_ids"]
                    padded_attention_mask[:local_bs, :local_seq] = probe_batches[i]["attention_mask"]
                    padded_position_ids[:local_bs, :local_seq] = probe_batches[i]["position_ids"]

                    # ✅ 立即销毁原始张量对象，极大地释放内存压力
                    probe_batches[i] = None

                # 防止 Dummy 区域全零导致底层 Softmax 出现 NaN
                padded_attention_mask[:, 0] = 1

                aligned_batches.append({
                    "input_ids": padded_input_ids,
                    "attention_mask": padded_attention_mask,
                    "position_ids": padded_position_ids
                })

            # 终极扫除，保障数据管道进入下一环节时不拖着旧变量
            del probe_batches

            return aligned_batches, metadata, local_count

        except Exception as e:
            self._brutal_crash_reporter("prepare_ig_probes", e)

    @trace_perf("ppo_actor.process_ig_rewards", category="compute")
    def process_ig_rewards(self, data: dict[str, Any], logprobs_list: list[torch.Tensor], metadata: Any) -> dict[
        str, Any]:
        try:
            ig_params = self.config.ig_reward_params
            valid_batch_sizes = metadata.get("valid_batch_sizes", [])
            cleaned_logprobs_list = []

            for i, logprobs in enumerate(logprobs_list):
                if i < len(valid_batch_sizes):
                    original_bs = valid_batch_sizes[i]
                    cleaned_logprobs_list.append(logprobs[:original_bs])
                else:
                    cleaned_logprobs_list.append(logprobs)

            from areal.utils.functional.functional import assign_ig_rewards
            return assign_ig_rewards(
                data=data,
                logprobs_list=cleaned_logprobs_list,
                metadata_dict=metadata,
                beta=ig_params.beta,
                use_peak_selection=getattr(ig_params, "use_peak_selection", False),
                use_watermark_selection=getattr(ig_params, "use_watermark_selection", False),  # [新增透传] 读取 config 传递给底层
                reward_mode=getattr(ig_params, "reward_mode", "prob")
            )
        except Exception as e:
            self._brutal_crash_reporter("process_ig_rewards", e)

    @trace_perf("ppo_actor.compute_advantages", category="compute")
    def compute_advantages(self, data: dict[str, Any]) -> dict[str, Any]:
        bs = data["input_ids"].shape[0]
        max_seqlen = data["input_ids"].shape[1]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor
            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        # Reward Scaling
        reward_score = data["rewards"]
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        if self.reward_norm:
            reward_score = self.reward_norm(reward_score)

        loss_mask = data["loss_mask"].float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        # Apply the mask to log probabilities.
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_no_eos_mask = seqlens == attn_mask.shape[1]

        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        rewards[batch_indices, seqlens - 1] = 0
        indices = torch.clip(seqlens - 2, min=0)

        if self.mask_no_eos_with_zero:
            rewards[batch_indices, indices] += torch.where(
                seq_no_eos_mask, 0, reward_score
            )
        else:
            rewards[batch_indices, indices] += reward_score

        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]

        # --- Dual-Stream GAE Computation ---

        # 1. Main Stream: Outcome + KL Reward
        adv_outcome = self._compute_general_advantage_estimation(
            rewards=rewards,
            values=values,
            loss_mask=loss_mask,
            seq_no_eos_mask=seq_no_eos_mask
        )

        # 2. IG Stream: Step-based Information Gain Reward
        adv_ig = torch.zeros_like(adv_outcome)

        if "token_level_rewards" in data:
            raw_step_rewards = data["token_level_rewards"]
            step_mask = (raw_step_rewards != 0).float()

            # 公式: sign(x) * ln(|x| + 1)
            # 作用: 完美保留正负号物理意义，将巨大差异的数值平滑压缩，且 0 值依然保持为 0
            scaled_step_rewards = torch.sign(raw_step_rewards) * torch.log1p(torch.abs(raw_step_rewards))

            len_penalties = data.get("token_level_len_penalties", torch.zeros_like(raw_step_rewards))
            scaled_step_rewards += len_penalties

            ig_lambda = 0.0
            if self.config.ig_reward_params:
                ig_lambda = self.config.ig_reward_params.lambda_val

            final_step_rewards = scaled_step_rewards * ig_lambda

            if "step_boundary_mask" in data:
                step_boundary_mask = data["step_boundary_mask"]
            else:
                step_boundary_mask = torch.ones_like(rewards, dtype=torch.float32)

            zeros_values = torch.zeros_like(values)
            adv_ig = self._compute_general_advantage_estimation(
                rewards=final_step_rewards,
                values=zeros_values,
                loss_mask=loss_mask,
                seq_no_eos_mask=seq_no_eos_mask,
                gae_mask=step_boundary_mask
            )

            # Debug logs
            if True:
                step_count = step_mask.sum().item()
                if step_count > 0:
                    ig_mean = final_step_rewards[step_mask.bool()].mean().item() if step_count > 0 else 0.0
                    ig_std = final_step_rewards[step_mask.bool()].std().item() if step_count > 0 else 0.0
                    ig_max = final_step_rewards[step_mask.bool()].max().item() if step_count > 0 else 0.0
                    ig_min = final_step_rewards[step_mask.bool()].min().item() if step_count > 0 else 0.0
                    logger.info(f"[IG-Debug] Step Rewards Non-Zero Count: {step_count}, Mean: {ig_mean}, Std: {ig_std}, Max: {ig_max}, Min: {ig_min}")
                else:
                    logger.info(
                        f"[IG-Debug] Step Rewards Non-Zero Count: {step_count}")

            rewards += final_step_rewards

            # -----------------------------------------------------------------
            # [新增] 将 Step Reward 相关指标存入 data 字典，供 ppo_update 收集
            # -----------------------------------------------------------------
            data["raw_step_rewards"] = raw_step_rewards
            data["final_step_rewards"] = final_step_rewards
            data["step_mask"] = step_mask

        # Final Advantages
        advantages = adv_outcome + adv_ig
        # advantages = adv_ig

        data["returns"] = advantages + values

        if self.adv_norm is not None:
            advantages = self.adv_norm(advantages, loss_mask)

        # Store data in the dict.
        data["advantages"] = advantages
        data["kl_rewards"] = kl_rewards
        data["tot_rewards"] = rewards
        data["loss_mask"] = loss_mask
        data["logprobs"] = old_logp

        return data

    def _compute_general_advantage_estimation(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            loss_mask: torch.Tensor,
            seq_no_eos_mask: torch.Tensor,
            gae_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bs = rewards.shape[0]
        max_seqlen = rewards.shape[1]

        advantages_reversed = [
            torch.zeros(bs, dtype=torch.float32, device=values.device)
        ]
        lastgaelam = 0
        nextvalues = values[:, max_seqlen - 1] * seq_no_eos_mask

        for t in reversed(range(max_seqlen - 1)):
            delta = rewards[:, t] + self.discount * nextvalues - values[:, t]

            if gae_mask is not None:
                lastgaelam = lastgaelam * gae_mask[:, t]

            newgaelam = delta + self.discount * self.gae_lambda * lastgaelam

            mask = loss_mask[:, t]
            nextvalues = nextvalues * (1 - mask) + values[:, t] * mask
            lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
            advantages_reversed.append(lastgaelam)

        return torch.stack(advantages_reversed[::-1], dim=1)

    @trace_perf("ppo_actor.ppo_update", category="compute")
    @stats_tracker.scope_func_wrapper("ppo_actor")
    def ppo_update(self, data: dict[str, Any]) -> None:
        attn_mask = data["attention_mask"]
        loss_mask = data["loss_mask"]
        reward_score = data["rewards"]
        seqlens = attn_mask.sum(-1)

        ########## Logging code starts ##########
        result_denominators = {
            "correct_n_seqs": (reward_score > 0).bool(),
            "incorrect_n_seqs": (reward_score <= 0).bool(),
        }
        if self.config.log_agent_stats:
            if "begin_of_trajectory" not in data:
                raise RuntimeError(
                    "'begin_of_trajectory' is expected to log agent statistics"
                )
            if len(self.config.log_agent_stats_keys) == 0:
                raise RuntimeError(
                    "`log_agent_stats_keys` should not be empty when log_agent_stats=True"
                )
            agent_denominator = (data["begin_of_trajectory"] > 0).bool()
            result_denominators["agent"] = agent_denominator
        global_denominators = dict(
            n_seqs=torch.ones_like(reward_score, dtype=torch.bool),
            n_tokens=torch.ones_like(loss_mask, dtype=torch.bool),
            n_valid_tokens=loss_mask.bool(),
            **result_denominators,
        )

        # -----------------------------------------------------------------
        # [新增] 注册自定义的分母：n_step_tokens，只包含有效触发了 Step 奖励的 Token 位置
        # -----------------------------------------------------------------
        if "step_mask" in data:
            global_denominators["n_step_tokens"] = data["step_mask"].bool()

        stats_tracker.denominator(**global_denominators)
        stats_tracker.stat(
            correct_seq_len=seqlens.float(), denominator="correct_n_seqs"
        )
        stats_tracker.stat(
            incorrect_seq_len=seqlens.float(), denominator="incorrect_n_seqs"
        )

        stats = dict(
            advantages=data["advantages"],
            kl_rewards=data["kl_rewards"],
            final_reward=data["tot_rewards"],
        )
        stats_tracker.stat(**stats, denominator="n_valid_tokens")

        # -----------------------------------------------------------------
        # [新增] 正式提交 Step Reward 统计指标
        # -----------------------------------------------------------------
        if "final_step_rewards" in data:
            # 1. 平均单步奖励 (除以所有出现 Step 标记的 Token 数量，保证均值不被 0 稀释)
            stats_tracker.stat(
                raw_step_reward=data["raw_step_rewards"],
                final_step_reward=data["final_step_rewards"],
                denominator="n_step_tokens"
            )
            # 2. 序列总累计奖励 (将每条序列上的所有步骤奖励相加，然后除以 Batch Size 即 n_seqs)
            seq_step_rewards = data["final_step_rewards"].sum(dim=-1)
            seq_step_counts = data["step_mask"].sum(dim=-1)
            stats_tracker.stat(
                seq_step_reward_sum=seq_step_rewards,
                seq_step_count=seq_step_counts,
                denominator="n_seqs"
            )
            # 3. 计算并记录全局 Step Reward 的标准差 (Std)
            step_mask_bool = data["step_mask"].bool()
            valid_final_rewards = data["final_step_rewards"][step_mask_bool]
            valid_raw_rewards = data["raw_step_rewards"][step_mask_bool]

            # numel() > 1 保护，防止只有1个或0个step时求标准差返回 NaN
            final_std = valid_final_rewards.std() if valid_final_rewards.numel() > 1 else torch.tensor(0.0)
            raw_std = valid_raw_rewards.std() if valid_raw_rewards.numel() > 1 else torch.tensor(0.0)

            stats_tracker.scalar(
                final_step_reward_std=final_std.item(),
                raw_step_reward_std=raw_std.item()
            )

        prompt_lens = data["attention_mask"].sum(-1) - data["loss_mask"].sum(-1)
        seq_stats = dict(
            no_eos_ratios=(seqlens == attn_mask.shape[-1]).float(),
            task_reward=reward_score.float(),
            prompt_len=prompt_lens.float(),
            seq_len=seqlens.float(),
        )

        stats_tracker.stat(**seq_stats, denominator="n_seqs")
        scalars = dict(
            mask_no_eos_with_zero=self.config.mask_no_eos_with_zero,
            eps_clip=self.config.eps_clip,
        )
        if self.config.c_clip is not None:
            scalars["c_clip"] = self.config.c_clip
            scalars["use_dual_clip"] = 1
        else:
            scalars["use_dual_clip"] = 0
        if self.config.behave_imp_weight_cap is not None:
            scalars["behave_imp_weight_cap"] = self.config.behave_imp_weight_cap
        stats_tracker.scalar(**scalars)

        if self.config.log_agent_stats:
            stats_tracker.stat(
                **{k: data[k].float() for k in self.config.log_agent_stats_keys},
                denominator="agent",
            )
        ########## Logging code ends ##########

        # Pop keys that are no longer needed after advantage computation
        # Note: "versions" is kept if needed for approximation/metrics in loss function
        for key in ["rewards", "tot_rewards", "kl_rewards", "raw_step_rewards", "final_step_rewards", "step_mask"]:
            data.pop(key, None)
        # NOTE: calling engine.train() is critical to enabling gradient checkpointing
        self.engine.train()
        mb_inputs = split_padded_tensor_dict_into_mb_list(
            data,
            mb_spec=MicroBatchSpec(n_mbs=self.config.ppo_n_minibatches),
        )

        with stats_tracker.scope("update"):
            # Get current version for proximal approximation metrics
            current_version = self.engine.get_version()

            for mb in mb_inputs.mbs:
                train_stat = self.engine.train_batch(
                    mb,
                    loss_fn=functools.partial(
                        grpo_loss_fn,
                        eps_clip=self.config.eps_clip,
                        eps_clip_higher=self.config.eps_clip_higher,
                        c_clip=self.config.c_clip,
                        behave_imp_weight_cap=self.config.behave_imp_weight_cap,
                        m2_threshold=self.m2_threshold,
                        importance_sampling_level=self.config.importance_sampling_level,
                        current_version=current_version,
                        prox_logp_method=self.config.prox_logp_method,
                        use_sapo_loss=self.config.use_sapo_loss,
                        sapo_tau_pos=self.config.sapo_tau_pos,
                        sapo_tau_neg=self.config.sapo_tau_neg,
                        use_decoupled_loss=self.config.use_decoupled_loss,
                        behave_imp_weight_mode=self.config.behave_imp_weight_mode,
                    ),
                    loss_weight_fn=lambda x: x["loss_mask"].count_nonzero(),
                )
                stats_tracker.scalar(**train_stat)


class PPOActorController(TrainController):
    def compute_logp(self, *args, **kwargs):
        return self._custom_function_call("compute_logp", *args, **kwargs)

    def compute_advantages(self, *args, **kwargs):
        return self._custom_function_call("compute_advantages", *args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self._custom_function_call("ppo_update", *args, **kwargs)


def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behave_imp_weight_cap: float | None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_decoupled_loss: bool = False,
    behave_imp_weight_mode: str = "token_mask",
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
):
    """Loss function for actor step, all inputs should be splitted into
    pipeline micro batches, returns loss and logging stats."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    entropy = entropy.detach()

    # Resolve proximal log-probabilities based on method
    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # Apply M2PO masking if threshold is set
    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    # Use SAPO or PPO loss
    if use_sapo_loss:
        if use_decoupled_loss:
            raise ValueError(
                "SAPO is not compatible with `use_decoupled_loss=True`. "
                "Please set `actor.use_decoupled_loss=false` in your configuration."
            )
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=c_clip,
            proximal_logprobs=prox_logp,
            behave_imp_weight_cap=behave_imp_weight_cap,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
            behave_imp_weight_mode=behave_imp_weight_mode,
        )

    # Log training statistics
    stats_tracker.denominator(
        # NOTE: n_tokens must have shape [batch, seq] to match vocab stats.
        # Using torch.ones_like(loss_mask) ensures correct shape when this function is called
        # standalone (e.g., by tests), not just from ppo_update() which already
        # registers n_tokens.
        n_tokens=torch.ones_like(loss_mask, dtype=torch.bool, device=logprobs.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )

    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logp,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )

    if vocab_min_logits is not None and vocab_max_logits is not None:
        stats_tracker.stat(
            vocab_min_logits=vocab_min_logits,
            vocab_max_logits=vocab_max_logits,
            denominator="n_tokens",
        )

    # Log SAPO-specific statistics
    if use_sapo_loss:
        stats_tracker.stat(
            sapo_soft_gate=stat["sapo_soft_gate"],
            sapo_scaled_gate_pos=stat["sapo_scaled_gate_pos"],
            sapo_scaled_gate_neg=stat["sapo_scaled_gate_neg"],
            denominator="n_valid_tokens",
        )
    else:
        # Log clipping statistics (PPO only)
        clip_mask = stat["clip_mask"]
        clipped_new_logp = torch.where(clip_mask, logprobs.detach(), 0.0)
        clipped_old_logp = torch.where(clip_mask, old_logp, 0.0)
        stats_tracker.stat(
            clipped_new_logp=clipped_new_logp,
            clipped_old_logp=clipped_old_logp,
            denominator="clipped_tokens",
        )

    # Log proximal approximation metrics
    compute_logp_mask = stat.get("behave_mask", loss_mask)
    _log_proximal_approximation_stats(
        prox_logp_method=prox_logp_method,
        prox_logp_gt=prox_logp_gt,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
        compute_logp_mask=compute_logp_mask,
    )

    # Log version staleness metrics
    if "versions" in input_data and current_version is not None:
        version_metrics_mask = stat.get("behave_mask", loss_mask)
        _log_version_staleness_stats(
            versions=input_data["versions"],
            current_version=current_version,
            version_metrics_mask=version_metrics_mask,
        )

    return loss


# =============================================================================
# Core Functions
# =============================================================================


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, metrics).
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (should be detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor.

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


def _apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance.

    Args:
        old_logp: Behavior policy log-probabilities.
        prox_logp: Proximal policy log-probabilities.
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering.

    Returns:
        Updated loss mask with M2PO filtering applied.
    """
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask


# =============================================================================
# Logging Helper Functions
# =============================================================================

_EPSILON = 1e-8  # Small constant for numerical stability in relative error calculations


def _compute_importance_weight(
    logp_numerator: torch.Tensor,
    logp_denominator: torch.Tensor,
) -> torch.Tensor:
    """Compute importance weight as exp(logp_num - logp_denom)."""
    return torch.exp(logp_numerator - logp_denominator).float()


def _compute_approximation_errors(
    ground_truth: torch.Tensor,
    approximation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute error metrics between ground truth and approximation.

    Returns:
        Dictionary with abs_error, rel_error, and squared_error tensors.
    """
    diff = ground_truth - approximation
    abs_error = torch.abs(diff).float()
    rel_error = torch.abs(diff / (torch.abs(ground_truth) + _EPSILON)).float()
    squared_error = (diff * diff).float()
    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "squared_error": squared_error,
    }


def _tensor_scalar_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Compute scalar statistics (avg, max, min) for a tensor.

    Args:
        tensor: Input tensor to compute statistics on.

    Returns:
        Dictionary with avg, max, min as Python floats.
    """
    t = tensor.float()
    return {
        "avg": t.mean().item(),
        "max": t.max().item(),
        "min": t.min().item(),
    }


def _log_approximation_metrics_for_method(
    method_name: str,
    approx_logp: torch.Tensor,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    prox_logp_gt: torch.Tensor | None = None,
) -> None:
    """
    Log metrics for a single approximation method.

    Args:
        method_name: Name of the approximation method (e.g., "loglinear").
        approx_logp: Approximated proximal log-probabilities.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities.
        prox_logp_gt: Ground truth proximal logp, or None if unavailable.
    """
    # Compute importance weights from approximation
    behave_imp_weight = _compute_importance_weight(approx_logp, old_logp)
    importance_weight = _compute_importance_weight(logprobs, approx_logp)

    metrics = {
        f"{method_name}/approx_logp": approx_logp.float(),
        f"{method_name}/behave_imp_weight": behave_imp_weight,
        f"{method_name}/importance_weight": importance_weight,
    }

    # Add error metrics if ground truth is available
    if prox_logp_gt is not None:
        # Log-probability errors
        logp_errors = _compute_approximation_errors(prox_logp_gt, approx_logp)
        metrics.update(
            {
                f"{method_name}/abs_error": logp_errors["abs_error"],
                f"{method_name}/rel_error": logp_errors["rel_error"],
                f"{method_name}/squared_error": logp_errors["squared_error"],
            }
        )

        # Ground truth importance weights for comparison
        behave_imp_weight_gt = _compute_importance_weight(prox_logp_gt, old_logp)
        importance_weight_gt = _compute_importance_weight(logprobs, prox_logp_gt)

        # Importance weight errors
        behave_errors = _compute_approximation_errors(
            behave_imp_weight_gt, behave_imp_weight
        )
        imp_errors = _compute_approximation_errors(
            importance_weight_gt, importance_weight
        )

        metrics.update(
            {
                f"{method_name}/behave_imp_weight_abs_error": behave_errors[
                    "abs_error"
                ],
                f"{method_name}/behave_imp_weight_rel_error": behave_errors[
                    "rel_error"
                ],
                f"{method_name}/importance_weight_abs_error": imp_errors["abs_error"],
                f"{method_name}/importance_weight_rel_error": imp_errors["rel_error"],
            }
        )

    stats_tracker.stat(**metrics, denominator="n_valid_tokens")


def _log_proximal_approximation_stats(
    prox_logp_method: str,
    prox_logp_gt: torch.Tensor | None,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
    compute_logp_mask: torch.Tensor,
) -> None:
    """
    Log proximal policy approximation metrics based on the method.

    Args:
        prox_logp_method: The proximal logp method being used.
        prox_logp_gt: Ground truth proximal logp, or None if skipped.
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.
        compute_logp_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("compute_logp"):
        stats_tracker.denominator(n_valid_tokens=compute_logp_mask.bool())

        # Log ground truth when available
        if prox_logp_gt is not None:
            stats_tracker.stat(
                prox_logp_gt=prox_logp_gt.float(),
                denominator="n_valid_tokens",
            )

        # Skip if versions not available
        if versions is None or current_version is None:
            return

        if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
            # Loglinear mode: log approximation without error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=None,  # No ground truth in loglinear mode
                )

        elif prox_logp_method == PROX_LOGP_METHOD_METRICS and prox_logp_gt is not None:
            # Metrics mode: compute all methods with error metrics
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=None,  # Compute all methods
            )
            for method_name, approx_logp in approximations.items():
                _log_approximation_metrics_for_method(
                    method_name=method_name,
                    approx_logp=approx_logp,
                    old_logp=old_logp,
                    logprobs=logprobs,
                    prox_logp_gt=prox_logp_gt,
                )


def _log_version_staleness_stats(
    versions: torch.Tensor,
    current_version: int,
    version_metrics_mask: torch.Tensor,
) -> None:
    """
    Log sample staleness metrics based on policy versions.

    Args:
        versions: Per-token policy versions from rollout.
        current_version: Current training version.
        version_metrics_mask: Mask for valid tokens.
    """
    with stats_tracker.scope("version_stats"):
        stats_tracker.denominator(n_valid_tokens=version_metrics_mask.bool())

        v_proximal = current_version - 1
        v_theta = current_version
        v_behave = versions.float()

        # Filter to generated tokens only (version >= 0)
        valid_generated_mask = version_metrics_mask & (versions >= 0)

        if not valid_generated_mask.any():
            return

        # Compute staleness for valid tokens
        staleness_proximal = (v_proximal - v_behave)[valid_generated_mask]
        staleness_theta = (v_theta - v_behave)[valid_generated_mask]

        # Compute and log statistics
        proximal_stats = _tensor_scalar_stats(staleness_proximal)
        theta_stats = _tensor_scalar_stats(staleness_theta)

        stats_tracker.scalar(
            sample_staleness_proximal_avg=proximal_stats["avg"],
            sample_staleness_proximal_max=proximal_stats["max"],
            sample_staleness_proximal_min=proximal_stats["min"],
            sample_staleness_theta_avg=theta_stats["avg"],
            sample_staleness_theta_max=theta_stats["max"],
            sample_staleness_theta_min=theta_stats["min"],
            v_theta=v_theta,
            v_proximal=v_proximal,
            n_valid_generated_tokens=valid_generated_mask.sum().item(),
        )
