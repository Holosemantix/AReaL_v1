import uuid
import asyncio
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("RLVRWorkflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return list(input_ids)


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    return data["messages"]


class RLVRWorkflow(RolloutWorkflow):
    """Single-turn reward learning workflow supporting optional thinking tokens."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        rollout_stat_scope: str = "rollout",
        ig_reward_params=None,
        get_input_ids_fn: Callable[[Any, PreTrainedTokenizerFast, bool], list[int]]
        | str = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any]
        | str = default_data_extract_prompt_fn,
    ):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.rollout_stat_scope = rollout_stat_scope
        self.enable_thinking = enable_thinking
        self.ig_reward_params = ig_reward_params
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        # Support string paths for get_input_ids_fn
        if isinstance(get_input_ids_fn, str):
            get_input_ids_fn = import_from_string(get_input_ids_fn)
        self.get_input_ids_fn = get_input_ids_fn
        # Support string paths for data_extract_prompt_fn
        if isinstance(data_extract_prompt_fn, str):
            data_extract_prompt_fn = import_from_string(data_extract_prompt_fn)
        self.data_extract_prompt_fn = data_extract_prompt_fn

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """Decode completion and compute reward.

        Traces reward phase execution for SessionTracer. Decodes output tokens
        to string, calls async reward function, and logs metric to stats tracker.

        Returns
        -------
        float
            Reward value.
        """
        completions_str = self.tokenizer.decode(resp.output_tokens)
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            resp.input_tokens,
            resp.output_tokens,
            **task_data,
        )

        return reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """Generate one sample and compute its reward.

        Registers a new session for this sample, calls engine.agenerate,
        computes reward, and logs metrics. SessionTracer automatically
        tracks generate and reward phases via @trace_session decorators.

        Returns
        -------
        tuple[ModelResponse, float]
            Model response and reward value.
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)

        reward = await self._compute_rewards(resp, prompt_str, task_data)

        stats_tracker.get(workflow_context.stat_scope(self.rollout_stat_scope)).scalar(reward=reward)

        return resp, reward

    async def _compute_ig_probes(
            self,
            engine: InferenceEngine,
            resp: ModelResponse,
            data: dict[str, Any],
            outcome_reward: float
    ) -> dict[str, list[float]] | None:
        """
        利用热乎的 KV Cache，异步并发计算所有中间步的 IG Logprobs，并直接结算过程奖励。
        这样完全解耦了 Actor 进程，避免了多卡同步死锁和 FSDP Padding 带来的内存溢出。
        """
        if not self.ig_reward_params:
            return None

        gt_text = data.get("answer")
        if not gt_text:
            logger.warning(f"[IG-Rollout] Enabled but no 'answer' in data.")
            return None

        # 1. 构造 Target 后缀
        gt_ids = self.tokenizer.encode(gt_text, add_special_tokens=False)
        bridge_text = getattr(self.ig_reward_params, "bridge_text", "\n\nTherefore, the answer is ")
        bridge_ids = self.tokenizer.encode(bridge_text, add_special_tokens=False)
        boxed_first_half_ids = self.tokenizer.encode("\\boxed{", add_special_tokens=False)
        boxed_second_half_ids = self.tokenizer.encode("}", add_special_tokens=False)

        ig_gt_ids = bridge_ids + boxed_first_half_ids + gt_ids + boxed_second_half_ids
        target_len = len(gt_ids)
        tail_len = len(boxed_second_half_ids)

        # 2. 定位所有步骤的切分点 (根据 sep_token)
        sep_token = getattr(self.ig_reward_params, "sep_token", "\n")
        sep_ids = self.tokenizer.encode(sep_token, add_special_tokens=False)
        if not sep_ids:
            return None
        sep_id = sep_ids[-1]

        step_offsets = [i for i, token_id in enumerate(resp.output_tokens) if token_id == sep_id]
        if not step_offsets or step_offsets[-1] != len(resp.output_tokens) - 1:
            step_offsets.append(len(resp.output_tokens) - 1)

        # 3. 构造异步探测请求
        probe_input_ids_list = [resp.input_tokens + ig_gt_ids]  # p_0 (基准水位)
        for offset in step_offsets:
            probe_input_ids_list.append(resp.input_tokens + resp.output_tokens[:offset + 1] + ig_gt_ids)

        probe_reqs = []
        for p_ids in probe_input_ids_list:
            probe_gconfig = self.gconfig.new(n_samples=1)
            # 防止 max_tokens 计算溢出， max_tokens 设为 prompt 长度 + 新 token 长度 (1)
            probe_gconfig.max_tokens = len(p_ids) + 1
            probe_gconfig.max_new_tokens = 1
            probe_gconfig.temperature = 1.0
            probe_gconfig.prompt_logprobs = 1

            probe_req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=p_ids,
                gconfig=probe_gconfig,
                tokenizer=self.tokenizer,
                metadata={"ig_target_len": target_len,
                          "tail_len": tail_len}  # 透传目标长度给底层，用于精准拦截
            )
            probe_reqs.append(probe_req)

        # 并发执行探测！完全命中 vLLM/SGLang 的 Radix Cache
        probe_responses = []
        chunk_size = 1
        for i in range(0, len(probe_reqs), chunk_size):
            chunk_reqs = probe_reqs[i: i + chunk_size]
            chunk_tasks = [engine.agenerate(req) for req in chunk_reqs]
            chunk_resps = await asyncio.gather(*chunk_tasks)
            probe_responses.extend(chunk_resps)
            # print(f"len(step_offsets): {len(step_offsets)}, step_offsets: {step_offsets}\t"
            #       f"len(chunk_reqs[0].input_ids):{len(chunk_reqs[0].input_ids)}\t"
            #       f"chunk_reqs[0].metadata: {chunk_reqs[0].metadata}\t"
            #       f"len(chunk_resps[0].input_logprobs): {len(chunk_resps[0].input_logprobs)}\t"
            #       f"len(probe_reqs): {len(probe_reqs)}\t"
            #       f"gt_text: {gt_text}\t"
            #       f"sep_token: {sep_token}\t"
            #       f"sep_id: {sep_id}\t"
            #       f"ig_gt_ids: {ig_gt_ids}\t")

        # 4. 提取 Prompt Logprobs 中 Target 后缀部分的概率求和
        step_logprobs = []

        if tail_len > 0:
            pure_gt_ids = ig_gt_ids[-(target_len + tail_len):-tail_len]
        else:
            pure_gt_ids = ig_gt_ids[-target_len:]

        for p_resp in probe_responses:
            p_logprobs = getattr(p_resp, "input_logprobs", [])

            # [修改点 2]：精准切片，避开 bridge_ids 和 尾部的 }
            # 目标范围是倒数第 (target_len + tail_len) 个，到倒数第 tail_len 个
            if p_logprobs and len(p_logprobs) >= target_len + tail_len:
                if tail_len > 0:
                    gt_logprobs = p_logprobs[-(target_len + tail_len):-tail_len]
                else:
                    gt_logprobs = p_logprobs[-target_len:]

                step_sum = 0.0
                for i, p in enumerate(gt_logprobs):
                    expected_token_id = pure_gt_ids[i]
                    # print(f"expected_token_id: {expected_token_id}, p: {p}")
                    if isinstance(p, dict):
                        # ⚠️ 绝对不能用 list(p.values())[0] 盲提！
                        # 必须用 expected_token_id 去查字典！
                        if expected_token_id in p:
                            token_info = p[expected_token_id]
                            step_sum += token_info.logprob if hasattr(token_info, "logprob") else float(token_info)
                        else:
                            # 引擎没返回真实答案的概率(说明偏离太远排在Top-K开外)，赋予严厉惩罚
                            step_sum += -15.0
                    elif hasattr(p, "logprob"):
                        step_sum += p.logprob
                    else:
                        step_sum += float(p)
                step_logprobs.append(step_sum)
            else:
                step_logprobs.append(-100.0)

        # 5. 调用核心算法逻辑结算过程奖励
        from areal.utils.functional.functional import _compute_rewards_logic

        step_lengths = []
        last_offset = -1
        for offset in step_offsets:
            step_lengths.append(offset - last_offset)
            last_offset = offset

        calculated_rewards, calculated_penalties = _compute_rewards_logic(
            log_probs=step_logprobs,
            outcome_reward=outcome_reward,
            beta=getattr(self.ig_reward_params, "beta", 0.5),
            use_peak_selection=getattr(self.ig_reward_params, "use_peak_selection", False),
            use_watermark_selection=getattr(self.ig_reward_params, "use_watermark_selection", False),
            reward_mode=getattr(self.ig_reward_params, "reward_mode", "prob_diff"),
            gt_len=target_len,
            step_lengths=step_lengths
        )

        # 6. 将结算好的奖励散射 (Scatter) 对齐到原始序列的绝对位置上
        seq_len = len(resp.input_tokens) + len(resp.output_tokens)
        prompt_len = len(resp.input_tokens)

        token_level_rewards = [0.0] * seq_len
        token_level_len_penalties = [0.0] * seq_len
        step_reward_mask = [0.0] * seq_len
        step_boundary_mask = [1.0] * seq_len

        for r, p, offset in zip(calculated_rewards, calculated_penalties, step_offsets):
            abs_idx = prompt_len + offset
            reward_idx = abs_idx - 1  # 映射到步骤结束符的前一个 token (对齐 Actor 逻辑)

            if 0 <= reward_idx < seq_len:
                token_level_rewards[reward_idx] = r.item() if isinstance(r, torch.Tensor) else float(r)
                token_level_len_penalties[reward_idx] = p.item() if isinstance(p, torch.Tensor) else float(p)
                step_reward_mask[reward_idx] = 1.0
            if abs_idx < seq_len:
                step_boundary_mask[abs_idx] = 0.0

        return {
            "token_level_rewards": token_level_rewards,
            "token_level_len_penalties": token_level_len_penalties,
            "step_reward_mask": step_reward_mask,
            "step_boundary_mask": step_boundary_mask,
        }

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        # NOTE: load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        # Generate single response and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # Build result tensor dict with batch dim 1
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }

        # [新增配置控制]: 支持通过配置项无缝切换计算后端
        ig_backend = getattr(self.ig_reward_params, "compute_backend", "actor") if self.ig_reward_params else None

        if self.ig_reward_params:
            if ig_backend == "rollout":
                # [无缝切换] 走 Rollout 异步探测分支
                ig_results = await self._compute_ig_probes(engine, resp, data, reward)
                if ig_results:
                    for k, v in ig_results.items():
                        # 直接把 1D 数组转换为 float32 Tensor 塞进去，
                        # 最后的 unsqueeze(0) 会完美兼容 actor 里的 shape (bs, seq_len)
                        res[k] = torch.tensor(v, dtype=torch.float32)

            # --- 下面这部分无论哪个 backend 都需要计算（为了防崩溃以及向 Actor 提供结构数据） ---
            # 准备ig计算所需的数据：提取并Tokenize标准答案 (Ground Truth)
            # 1. 尝试从 data 中提取标准答案文本
            # 常见的 key 为 'answer', 'solution' 或 'ground_truth'
            # 如果你的数据集结构比较特殊，建议在 __init__ 中传入一个 data_extract_target_fn
            gt_text = data.get("answer")

            if gt_text is None:
                logger.warning(
                    f"IG Reward is enabled but no ground truth found in data. "
                    f"Available keys: {list(data.keys())}. "
                    f"IG calculation will likely fail or return 0."
                )
                gt_ids = []
            else:
                # 2. 编码标准答案
                # 注意：add_special_tokens=False，因为我们要把它拼接到推理链后面，
                # 不希望开头出现 BOS 或其他特殊 Token 干扰概率计算
                gt_ids = self.tokenizer.encode(gt_text, add_special_tokens=False)

            # 2. 识别 Step 边界 (通常是换行符 \n)
            # 假设 tokenizer 中 '\n' 的 id 是 newline_id
            # 注意：不同 tokenizer 的换行符 ID 可能不同，有的可能是 '\n' (Llama) 或 'Ċ' (GPT2/RoBERTa)
            sep_id = self.tokenizer.encode(self.ig_reward_params.sep_token, add_special_tokens=False)[-1]
            res["ig_sep_token_id"] = torch.tensor(sep_id, dtype=torch.int32)
            res["ig_pad_token_id"] = torch.tensor(self.tokenizer.pad_token_id, dtype=torch.int32)

            # 3. 准备 Bridge (连接词)
            # 这一步非常重要，必须让模型觉得“我要开始回答了”
            # 比如: "\n\n#### " 或者 "\n\nTherefore, the answer is "
            bridge_ids = self.tokenizer.encode(self.ig_reward_params.bridge_text, add_special_tokens=False)
            boxed_first_half_ids = self.tokenizer.encode("\\boxed{", add_special_tokens=False)
            boxed_second_half_ids = self.tokenizer.encode("}", add_special_tokens=False)

            # 4. 存入 res 字典
            # 下游 Actor/Critic 在计算 Advantage 时，会检查 'ig_gt_ids' 是否存在
            # 注意：此处直接转为 Tensor，数据类型需与 input_ids 保持一致 (int32)
            # return 语句会自动对其进行 unsqueeze(0) 处理，增加 batch 维度
            res["ig_gt_ids"] = torch.tensor(bridge_ids + boxed_first_half_ids + gt_ids + boxed_second_half_ids,
                                            dtype=torch.int32)
            # 同时存入长度，方便后续token切分定位处理
            res["ig_bridge_len"] = torch.tensor(len(bridge_ids + boxed_first_half_ids), dtype=torch.int32)
            res["ig_gt_len"] = torch.tensor(len(gt_ids), dtype=torch.int32)
            res["prompt_len"] = torch.tensor(len(resp.input_tokens), dtype=torch.int32)


        return {k: v.unsqueeze(0) for k, v in res.items()}
