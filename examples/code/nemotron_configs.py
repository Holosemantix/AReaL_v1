"""Configuration dataclasses for Nemotron code RL examples.

These live in a dedicated module so that RPC workers can deserialize them
(via importlib) when they are transmitted across the wire.  Dataclasses
defined directly in a ``__main__`` script cannot be imported by workers.
"""

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig, PPOActorConfig
from areal.workflow.model_scorer import ModelScorerConfig


@dataclass
class InfoGainRewardConfig:
    """Matches the shape expected by rl_trainer_qun_team / actor_qun_team."""

    mini_batch_size: int = field(default=128)
    max_tokens_per_forward: int = field(default=131072)
    compute_backend: str = field(default="rollout")
    bridge_text: str = field(default="")
    sep_token: str = field(default="\n")
    reward_mode: str = field(default="prob")
    beta: float = field(default=0.5)
    lambda_val: float = field(default=0.2)
    enable_varlen_packing: bool = field(default=True)
    use_watermark_selection: bool = field(default=False)
    use_peak_selection: bool = field(default=False)


@dataclass
class InfoGainPPOActorConfig(PPOActorConfig):
    ig_reward_params: InfoGainRewardConfig | None = field(default=None)


@dataclass
class CodeRewardConfig:
    """Budget for the competitive-coding execution reward.

    All values are tuned for H800 + 0.6B–14B Qwen3 rollouts.

    Note: with a string-based ``reward_fn`` (required for RPC serialization),
    the per-test fields below currently document the reward fn's defaults.
    To change them at training time, edit the defaults in
    ``areal.reward.code.competitive_exec.nemotron_competitive_reward_fn`` or
    inject overrides via the dataset loader (each sample's task_data flows
    into the reward fn as kwargs).
    """

    per_test_timeout: float = field(default=5.0)
    max_tests: int = field(default=15)
    memory_mb: int = field(default=2048)
    reward_timeout_seconds: float = field(default=90.0)
    # Multi-language support
    language: str = field(default="python")
    compile_timeout: float = field(default=30.0)
    java_xmx_mb: int = field(default=2048)
    # Toolchain probe at trainer startup
    probe_languages: list[str] = field(
        default_factory=lambda: ["python", "cpp", "java", "javascript"]
    )


@dataclass
class CodeGRPOConfig(GRPOConfig):
    actor: InfoGainPPOActorConfig = field(default_factory=InfoGainPPOActorConfig)
    code_reward: CodeRewardConfig = field(default_factory=CodeRewardConfig)
    model_scorer: ModelScorerConfig = field(default_factory=ModelScorerConfig)
