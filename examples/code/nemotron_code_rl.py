"""GRPO training on Nemotron-RL-coding-competitive_coding.

Pipeline:
  dataset   areal.dataset.code.nemotron_competitive
  reward    areal.reward.code.competitive_exec (subprocess sandbox, pass rate)
  workflow  areal.workflow.rlvr_qun_team.RLVRWorkflow (with extended reward timeout)

Note:
  ``reward_fn`` and ``workflow`` are passed as import-path strings so they
  survive JSON-RPC serialization to remote workers.  The default kwargs of
  ``areal.reward.code.nemotron_competitive_reward_fn`` match the
  ``code_reward`` fields in the YAML template.
"""

import sys
from dataclasses import asdict, is_dataclass

from examples.code.nemotron_configs import CodeGRPOConfig

from areal.api.cli_args import load_expr_config
from areal.dataset.get_datasets_qun_team import get_multi_custom_dataset
from areal.trainer.rl_trainer_qun_team import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, CodeGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Probe sandbox toolchain early so missing g++/java/node show up in the
    # trainer log before any rollouts happen.
    from areal.reward.code import assert_toolchain_or_warn

    assert_toolchain_or_warn(list(config.code_reward.probe_languages))

    if config.actor.ig_reward_params is not None:
        config.actor.ig_reward_params.sep_token_id = tokenizer.encode(
            config.actor.ig_reward_params.sep_token, add_special_tokens=False
        )
        config.actor.ig_reward_params.pad_token_id = tokenizer.pad_token_id

    train_dataset_map = get_multi_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    assert len(train_dataset_map) == 1, f"{len(train_dataset_map)} != 1"
    train_dataset = next(iter(train_dataset_map.values()))

    valid_dataset_map: dict = {}
    if config.valid_dataset is not None and config.valid_dataset.path:
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
            reward_fn="areal.reward.code.nemotron_competitive_reward_fn",
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
            enable_thinking=config.gconfig.enable_thinking,
            reward_timeout_seconds=config.code_reward.reward_timeout_seconds,
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
            workflow="areal.workflow.rlvr_qun_team.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr_qun_team.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs_map,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
