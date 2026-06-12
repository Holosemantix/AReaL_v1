import sys
from dataclasses import asdict, dataclass, field, is_dataclass

from areal.api.cli_args import GRPOConfig, PPOActorConfig, load_expr_config
from areal.dataset.get_datasets_qun_team import get_multi_custom_dataset
from areal.reward.math_verify_qun_team import math_verify_reward_fn
from areal.trainer.rl_trainer_qun_team import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.workflow.model_scorer import ModelScorerConfig
from areal.workflow.rlvr_qun_team import RLVRWorkflow


@dataclass
class InfoGainRewardConfig:
    """Configuration for information gain reward."""

    mini_batch_size: int = field(
        default=128,
        metadata={"help": "mini_batch_size for each sample ig reward calculation."},
    )
    max_tokens_per_forward: int = field(default=131072)
    compute_backend: str = field(
        default="rollout",
        metadata={
            "help": "choise of compute logprobs backends in ['rollout', 'actor']"
        },
    )
    bridge_text: str = field(default="")
    sep_token: str = field(default="\n")
    reward_mode: str = field(default="prob")
    beta: float = field(
        default=0.5,
        metadata={"help": "weight of future ig."},
    )
    lambda_val: float = field(
        default=0.2,
        metadata={"help": "weight of total ig."},
    )
    enable_varlen_packing: bool = field(
        default=True,
        metadata={"help": "enable_varlen_packing."},
    )
    use_watermark_selection: bool = field(
        default=False,
        metadata={"help": "use_watermark_selection."},
    )
    use_peak_selection: bool = field(
        default=False,
        metadata={"help": "Peak selection switch."},
    )


@dataclass
class ShortestCorrectRewardConfig:
    """Group-relative length penalty for correct math rollouts."""

    enabled: bool = field(default=False)
    alpha: float = field(default=0.1)
    reward_threshold: float = field(default=1.0)
    min_correct: int = field(default=2)
    group_size: int | None = field(default=None)
    normalize_by_shortest: bool = field(default=True)
    max_penalty: float | None = field(default=1.0)
    min_shortest_len: int = field(default=1)


@dataclass
class AdaptiveLengthRewardConfig:
    """Group solve-rate adaptive length penalty for math rollouts."""

    enabled: bool = field(default=False)
    alpha: float = field(default=0.05)
    reward_threshold: float = field(default=1.0)
    min_correct: int = field(default=2)
    group_size: int | None = field(default=None)
    min_solve_rate: float = field(default=0.5)
    max_solve_rate: float = field(default=1.0)
    target_quantile: float = field(default=0.25)
    min_target_len: int = field(default=2048)
    normalize_by_target: bool = field(default=True)
    max_penalty: float | None = field(default=0.1)
    correct_only: bool = field(default=True)


@dataclass
class InfoGainPPOActorConfig(PPOActorConfig):
    ig_reward_params: InfoGainRewardConfig | None = field(
        default=None,
        metadata={"help": "ig_reward configuration."},
    )
    shortest_correct_reward: ShortestCorrectRewardConfig = field(
        default_factory=ShortestCorrectRewardConfig,
        metadata={"help": "Group-relative shortest-correct length reward."},
    )
    adaptive_length_reward: AdaptiveLengthRewardConfig = field(
        default_factory=AdaptiveLengthRewardConfig,
        metadata={"help": "Group solve-rate adaptive length reward."},
    )


@dataclass
class InfoGainGRPOConfig(GRPOConfig):
    actor: InfoGainPPOActorConfig = field(default_factory=InfoGainPPOActorConfig)
    model_scorer: ModelScorerConfig = field(default_factory=ModelScorerConfig)


def main(args):
    config, _ = load_expr_config(args, InfoGainGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    if config.actor.ig_reward_params is not None:
        config.actor.ig_reward_params.sep_token_id = tokenizer.encode(
            config.actor.ig_reward_params.sep_token, add_special_tokens=False
        )
        config.actor.ig_reward_params.pad_token_id = tokenizer.pad_token_id

    train_dataset = get_multi_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    assert len(train_dataset) == 1, f"{len(train_dataset)} != 1"
    for _, dataset in train_dataset.items():
        train_dataset = dataset

    valid_dataset_map = get_multi_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset_map,
    ) as trainer:
        workflow_kwargs = dict(
            reward_fn=math_verify_reward_fn,
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=config.gconfig.enable_thinking,
            model_scorer=asdict(config.model_scorer)
            if is_dataclass(config.model_scorer)
            else config.model_scorer,
        )

        eval_workflow_kwargs_map = {}
        for dataset_name in valid_dataset_map.keys():
            eval_workflow_kwargs = workflow_kwargs.copy()
            eval_workflow_kwargs["gconfig"] = config.eval_gconfig
            eval_workflow_kwargs["rollout_stat_scope"] = f"eval-rollout/{dataset_name}"
            eval_workflow_kwargs_map[dataset_name] = eval_workflow_kwargs

        workflow_kwargs["ig_reward_params"] = config.actor.ig_reward_params
        trainer.train(
            workflow=RLVRWorkflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=RLVRWorkflow,
            eval_workflow_kwargs=eval_workflow_kwargs_map,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
