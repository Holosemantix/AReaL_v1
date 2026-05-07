import asyncio
import json
import os
import re
from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any

from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.utils import logging

logger = logging.getLogger("ModelScorer")


@dataclass
class ModelScorerInputConfig:
    mode: str = "prompt_completion"
    prompt_tail_tokens: int | None = None
    completion_tail_tokens: int | None = None
    include_answer: bool = True
    include_tests: bool = False
    max_field_chars: int = 4096
    system_prompt: str = (
        "You are a strict reward model. Grade whether the completion solves the "
        "task. Return only JSON with a numeric score in [0, 1], e.g. "
        '{"score": 0.75}.'
    )


@dataclass
class ModelScorerConfig:
    enabled: bool = False
    backend: str = "external"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: float = 60.0
    max_concurrent_requests: int = 32
    temperature: float = 0.0
    max_tokens: int = 32
    combine_mode: str = "blend"
    weight: float = 0.5
    min_score: float = 0.0
    max_score: float = 1.0
    fallback_score: float = 0.0
    input: ModelScorerInputConfig = field(default_factory=ModelScorerInputConfig)


def normalize_model_scorer_config(
    config: ModelScorerConfig | dict | None,
) -> ModelScorerConfig:
    if config is None:
        return ModelScorerConfig()
    if isinstance(config, ModelScorerConfig):
        return config
    if is_dataclass(config):
        config = asdict(config)
    if not isinstance(config, dict):
        raise TypeError(f"Unsupported model_scorer config type: {type(config)}")
    cfg = dict(config)
    input_cfg = cfg.get("input")
    if isinstance(input_cfg, ModelScorerInputConfig):
        pass
    elif input_cfg is None:
        cfg["input"] = ModelScorerInputConfig()
    elif is_dataclass(input_cfg):
        cfg["input"] = ModelScorerInputConfig(**asdict(input_cfg))
    elif isinstance(input_cfg, dict):
        cfg["input"] = ModelScorerInputConfig(**input_cfg)
    else:
        raise TypeError(f"Unsupported model_scorer.input type: {type(input_cfg)}")
    return ModelScorerConfig(**cfg)


class ModelScorer:
    def __init__(
        self,
        config: ModelScorerConfig | dict | None,
        *,
        tokenizer: PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters,
    ):
        self.config = normalize_model_scorer_config(config)
        self.tokenizer = tokenizer
        self.gconfig = gconfig
        self._semaphore = asyncio.Semaphore(
            max(1, int(self.config.max_concurrent_requests))
        )
        self._external_client = None

        if self.config.enabled and self.config.backend not in {"external", "actor"}:
            raise ValueError(
                "model_scorer.backend must be one of {'external', 'actor'}, "
                f"got {self.config.backend!r}"
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def combine(self, rule_reward: float, model_score: float) -> float:
        mode = self.config.combine_mode
        weight = self.config.weight
        if mode == "rule_only":
            return rule_reward
        if mode == "replace":
            return model_score
        if mode == "add":
            return rule_reward + weight * model_score
        if mode == "blend":
            return (1.0 - weight) * rule_reward + weight * model_score
        raise ValueError(
            "model_scorer.combine_mode must be one of "
            "{'rule_only', 'replace', 'add', 'blend'}, "
            f"got {mode!r}"
        )

    async def score_and_combine(
        self,
        *,
        engine: InferenceEngine,
        rule_reward: float,
        resp: ModelResponse,
        prompt_str: str,
        completion_str: str,
        task_data: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        if not self.enabled:
            return rule_reward, {}

        async with self._semaphore:
            try:
                if self.config.backend == "external":
                    model_score = await self._score_external(
                        resp=resp,
                        prompt_str=prompt_str,
                        completion_str=completion_str,
                        task_data=task_data,
                    )
                else:
                    model_score = await self._score_actor(
                        engine=engine,
                        resp=resp,
                        prompt_str=prompt_str,
                        completion_str=completion_str,
                        task_data=task_data,
                    )
                error = 0.0
            except Exception:
                logger.warning("Model scorer failed; using fallback score.", exc_info=True)
                model_score = self.config.fallback_score
                error = 1.0

        model_score = self._clamp_score(model_score)
        final_reward = self.combine(rule_reward, model_score)
        return final_reward, {
            "model_scorer/rule_reward": rule_reward,
            "model_scorer/model_score": model_score,
            "model_scorer/final_reward": final_reward,
            "model_scorer/error": error,
        }

    async def _score_external(
        self,
        *,
        resp: ModelResponse,
        prompt_str: str,
        completion_str: str,
        task_data: dict[str, Any],
    ) -> float:
        from openai import AsyncOpenAI

        base_url = self.config.base_url or os.getenv("MODEL_SCORER_BASE_URL")
        api_key = self.config.api_key or os.getenv("MODEL_SCORER_API_KEY")
        model = self.config.model or os.getenv("MODEL_SCORER_MODEL")
        if not base_url or not model:
            raise ValueError(
                "External model scorer requires model_scorer.base_url and "
                "model_scorer.model, or MODEL_SCORER_BASE_URL/MODEL_SCORER_MODEL."
            )
        if self._external_client is None:
            self._external_client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "dummy",
                max_retries=0,
                timeout=self.config.timeout_seconds,
            )
        completion = await self._external_client.chat.completions.create(
            model=model,
            messages=self._build_messages(
                resp=resp,
                prompt_str=prompt_str,
                completion_str=completion_str,
                task_data=task_data,
            ),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout_seconds,
        )
        content = completion.choices[0].message.content or ""
        return self._parse_score(content)

    async def _score_actor(
        self,
        *,
        engine: InferenceEngine,
        resp: ModelResponse,
        prompt_str: str,
        completion_str: str,
        task_data: dict[str, Any],
    ) -> float:
        messages = self._build_messages(
            resp=resp,
            prompt_str=prompt_str,
            completion_str=completion_str,
            task_data=task_data,
        )
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = list(input_ids)
        gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.config.max_tokens,
            max_tokens=len(input_ids) + self.config.max_tokens,
            temperature=self.config.temperature,
            greedy=self.config.temperature <= 0,
        )
        judge_resp = await engine.agenerate(
            ModelRequest(
                rid=f"model-scorer-{id(resp)}",
                input_ids=input_ids,
                gconfig=gconfig,
                tokenizer=self.tokenizer,
            )
        )
        content = self.tokenizer.decode(judge_resp.output_tokens)
        return self._parse_score(content)

    def _build_messages(
        self,
        *,
        resp: ModelResponse,
        prompt_str: str,
        completion_str: str,
        task_data: dict[str, Any],
    ) -> list[dict[str, str]]:
        prompt_text = self._select_text(
            resp.input_tokens, prompt_str, self.config.input.prompt_tail_tokens
        )
        completion_text = self._select_text(
            resp.output_tokens,
            completion_str,
            self.config.input.completion_tail_tokens,
        )
        if self.config.input.mode == "completion_only":
            prompt_text = ""
        elif self.config.input.mode != "prompt_completion":
            raise ValueError(
                "model_scorer.input.mode must be 'prompt_completion' or "
                f"'completion_only', got {self.config.input.mode!r}"
            )

        sections = []
        if prompt_text:
            sections.append(("Task prompt", prompt_text))
        sections.append(("Model completion", completion_text))
        if self.config.input.include_answer and "answer" in task_data:
            sections.append(("Reference answer", str(task_data["answer"])))
        if self.config.input.include_tests:
            tests = self._format_tests(task_data)
            if tests:
                sections.append(("Tests", tests))

        body = "\n\n".join(
            f"## {title}\n{self._truncate(text)}" for title, text in sections
        )
        body += (
            "\n\nGrade the completion as a scalar score from 0 to 1. "
            'Return only JSON like {"score": 0.0}.'
        )
        return [
            {"role": "system", "content": self.config.input.system_prompt},
            {"role": "user", "content": body},
        ]

    def _select_text(
        self,
        token_ids: list[int],
        full_text: str,
        tail_tokens: int | None,
    ) -> str:
        if tail_tokens is None or tail_tokens <= 0 or len(token_ids) <= tail_tokens:
            return full_text
        return self.tokenizer.decode(token_ids[-tail_tokens:])

    def _format_tests(self, task_data: dict[str, Any]) -> str:
        inputs = task_data.get("test_inputs")
        outputs = task_data.get("test_outputs")
        if not inputs or not outputs:
            return ""
        pairs = []
        for idx, (test_input, test_output) in enumerate(zip(inputs[:3], outputs[:3])):
            pairs.append(
                f"Test {idx + 1}\nInput:\n{test_input}\nExpected output:\n{test_output}"
            )
        return "\n\n".join(pairs)

    def _truncate(self, text: str) -> str:
        max_chars = self.config.input.max_field_chars
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n...[truncated]...\n" + text[-half:]

    def _parse_score(self, text: str) -> float:
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "score" in parsed:
                return float(parsed["score"])
            if isinstance(parsed, int | float):
                return float(parsed)
        except json.JSONDecodeError:
            pass

        match = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', stripped)
        if match:
            return float(match.group(1))
        match = re.search(r"-?\d+(?:\.\d+)?", stripped)
        if match:
            return float(match.group(0))
        raise ValueError(f"Could not parse model scorer output: {text!r}")

    def _clamp_score(self, score: float) -> float:
        return max(self.config.min_score, min(self.config.max_score, float(score)))
