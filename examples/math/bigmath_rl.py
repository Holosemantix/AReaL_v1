import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GRPOConfig, load_expr_config, PPOActorConfig, GenerationHyperparameters
from areal.dataset.get_datasets_qun_team import get_multi_custom_dataset
from areal.trainer.rl_trainer_qun_team import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.reward.math_verify_qun_team import math_verify_reward_fn
from areal.workflow.rlvr_qun_team import RLVRWorkflow, default_get_input_ids_fn, default_data_extract_prompt_fn


@dataclass
class InfoGainRewardConfig:
    """Configuration for information gain reward."""
    mini_batch_size: int = field(
        default=128,
        metadata={
            "help": "mini_batch_size for each sample ig reward calculation."
        },
    )
    max_tokens_per_forward: int = field(default=131072)
    compute_backend: str = field(
        default="rollout",
        metadata={
            "help": "choise of compute logprobs backends in ['rollout', 'actor']"
        },)
    bridge_text: str = field(default="")
    sep_token: str = field(default="\n")
    reward_mode: str = field(default="prob")
    beta: float = field(
        default=0.5,
        metadata={
            "help": "weight of future ig."
        },
    )
    lambda_val: float = field(
        default=0.2,
        metadata={
            "help": "weight of total ig."
        },
    )
    enable_varlen_packing: bool = field(
        default=True,
        metadata={
            "help": "enable_varlen_packing."
        },
    )
    use_watermark_selection: bool = field(
        default=False,
        metadata={
            "help": "use_watermark_selection."
        },
    )
    use_peak_selection: bool = field(
        default=False,
        metadata={
            "help": "Peak selection switch."
        },
    )


@dataclass
class InfoGainPPOActorConfig(PPOActorConfig):
    ig_reward_params: InfoGainRewardConfig | None = field(
        default=None,
        metadata={"help": "ig_reward configuration."},
    )


@dataclass
class InfoGainGRPOConfig(GRPOConfig):
    actor: InfoGainPPOActorConfig = field(default_factory=InfoGainPPOActorConfig)


def main(args):
    config, _ = load_expr_config(args, InfoGainGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    if config.actor.ig_reward_params is not None:
        config.actor.ig_reward_params.sep_token_id = tokenizer.encode(config.actor.ig_reward_params.sep_token, add_special_tokens=False)
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
        )

        eval_workflow_kwargs_map = {}
        for dataset_name in valid_dataset_map.keys():
            eval_workflow_kwargs = workflow_kwargs.copy()
            eval_workflow_kwargs["gconfig"] = config.eval_gconfig
            eval_workflow_kwargs["rollout_stat_scope"] = f"eval-rollout/{dataset_name}"
            eval_workflow_kwargs_map[dataset_name] = eval_workflow_kwargs

        workflow_kwargs['ig_reward_params'] = config.actor.ig_reward_params
        trainer.train(
            workflow=RLVRWorkflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=RLVRWorkflow,
            eval_workflow_kwargs=eval_workflow_kwargs_map,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
