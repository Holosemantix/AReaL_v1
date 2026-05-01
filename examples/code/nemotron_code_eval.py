"""Standalone RLVR evaluation for Nemotron competitive coding data."""

import sys

from examples.code.nemotron_configs import CodeGRPOConfig

from areal.api.cli_args import load_expr_config
from areal.dataset.get_datasets_qun_team import get_multi_custom_dataset
from areal.reward.code import assert_toolchain_or_warn
from areal.trainer.rl_eval_qun_team import RLEvaluator
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, CodeGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    assert_toolchain_or_warn(list(config.code_reward.probe_languages))

    if config.valid_dataset is None or not config.valid_dataset.path:
        raise ValueError("valid_dataset.path must be set for standalone evaluation.")

    valid_dataset_map = get_multi_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    with RLEvaluator(config, valid_dataset=valid_dataset_map) as evaluator:
        eval_workflow_kwargs_map = {}
        for dataset_name in valid_dataset_map.keys():
            eval_workflow_kwargs_map[dataset_name] = dict(
                reward_fn="areal.reward.code.nemotron_competitive_reward_fn",
                gconfig=config.eval_gconfig,
                tokenizer=config.tokenizer_path,
                enable_thinking=config.gconfig.enable_thinking,
                reward_timeout_seconds=config.code_reward.reward_timeout_seconds,
                rollout_stat_scope=f"eval-rollout/{dataset_name}",
            )

        evaluator.evaluate(
            eval_workflow="areal.workflow.rlvr_qun_team.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs_map,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
