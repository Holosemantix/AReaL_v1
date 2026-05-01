from __future__ import annotations

import json
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Any

from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode, AllocationType
from areal.api.cli_args import (
    InferenceEngineConfig,
    PPOConfig,
    SGLangConfig,
    ValidDatasetConfig,
    vLLMConfig,
)
from areal.api.io_struct import FinetuneSpec
from areal.api.scheduler_api import Scheduler
from areal.api.workflow_api import WorkflowLike
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra import LocalScheduler, RayScheduler, RolloutController, SlurmScheduler
from areal.utils import logging, perf_tracer, seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.environ import is_single_controller
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.perf_tracer import Category
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("RLEvaluator")


class RLEvaluator:
    """Standalone RLVR evaluator using the same rollout path as PPOTrainer."""

    def __init__(
        self,
        config: PPOConfig,
        valid_dataset: Dataset | dict[str, Dataset] | None = None,
    ):
        if not is_single_controller():
            raise NotImplementedError(
                "RLEvaluator must run as a single controller process. "
                "Use `python3 -m <eval_entrypoint> scheduler.type=local|ray` "
                "instead of launching it through areal.infra.launcher.*."
            )

        logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        self.config = config
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )
        seeding.set_random_seed(config.seed, key="eval")

        self.scheduler = self._init_scheduler()
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)
        self._validate_eval_allocation_mode()
        self._validate_cfg()

        self.valid_dataset = valid_dataset
        self.valid_dataloader = self._init_valid_dataloader(valid_dataset)
        ft_spec = self._build_finetune_spec()
        self.stats_logger = StatsLogger(config, ft_spec)

        self.eval_rollout = self._init_eval_rollout(config.rollout)

    def evaluate(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs: dict[str, Any] | dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        if self.valid_dataloader is None:
            raise ValueError("valid_dataset is required for standalone evaluation.")

        with (
            stats_tracker.record_timing("eval"),
            perf_tracer.trace_scope(
                "eval.rollout",
                category=Category.COMPUTE,
                args={"global_step": 0},
            ),
        ):
            self._evaluate_fn(eval_workflow, eval_workflow_kwargs)

        stats = self.eval_rollout.export_stats()
        stats.update(stats_tracker.export_all(reset=True))
        self.stats_logger.commit(epoch=0, step=0, global_step=0, data=stats)
        self._save_log_metrics(metrics=stats, epoch=0, epoch_step=0, global_step=0)
        return stats

    def close(self) -> None:
        self.stats_logger.close()
        self.eval_rollout.destroy()
        perf_tracer.save(force=True)

    def _init_scheduler(self) -> Scheduler:
        cfg = self.config.scheduler
        if cfg.type == "local":
            return LocalScheduler(exp_config=self.config)
        if cfg.type == "ray":
            import ray

            if not ray.is_initialized():
                ray.init(address="auto")
            return RayScheduler(exp_config=self.config)
        if cfg.type == "slurm":
            return SlurmScheduler(exp_config=self.config)
        raise NotImplementedError(f"Unknown scheduler type: {cfg.type}")

    def _init_valid_dataloader(
        self, valid_dataset: Dataset | dict[str, Dataset] | None
    ) -> StatefulDataLoader | dict[str, StatefulDataLoader] | None:
        if self.config.valid_dataset is None or valid_dataset is None:
            return None

        if isinstance(valid_dataset, dict):
            valid_dataloader = {}
            for dataset_name, single_valid_dataset in valid_dataset.items():
                valid_dataloader[dataset_name] = self._create_dataloader(
                    single_valid_dataset,
                    dataset_config=self.config.valid_dataset,
                    rank=0,
                    world_size=1,
                )
                logger.info("dataset: %s", dataset_name)
            return valid_dataloader

        return self._create_dataloader(
            valid_dataset,
            dataset_config=self.config.valid_dataset,
            rank=0,
            world_size=1,
        )

    def _create_dataloader(
        self,
        dataset: Dataset,
        dataset_config: ValidDatasetConfig,
        rank: int,
        world_size: int,
    ) -> StatefulDataLoader:
        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
        )

    def _build_finetune_spec(self) -> FinetuneSpec:
        batch_size = (
            self.config.valid_dataset.batch_size
            if self.config.valid_dataset is not None
            else 1
        )
        if isinstance(self.valid_dataloader, dict):
            steps = sum(len(dataloader) for dataloader in self.valid_dataloader.values())
        elif self.valid_dataloader is not None:
            steps = len(self.valid_dataloader)
        else:
            steps = 1
        steps = max(steps, 1)
        return FinetuneSpec(
            total_train_epochs=1,
            dataset_size=steps * batch_size,
            train_batch_size=batch_size,
        )

    def _init_eval_rollout(
        self, rollout_config: InferenceEngineConfig
    ) -> RolloutController:
        config = deepcopy(rollout_config)
        config.max_head_offpolicyness = int(1e12)

        if self.allocation_mode.gen_backend == "sglang":
            if self.config.rollout.return_routed_experts:
                self.config.sglang.enable_return_routed_experts = True
            engine_cls = RemoteSGLangEngine
            server_args = SGLangConfig.build_args(
                sglang_config=self.config.sglang,
                tp_size=self.allocation_mode.gen.tp_size,
                base_gpu_id=0,
            )
        elif self.allocation_mode.gen_backend == "vllm":
            engine_cls = RemotevLLMEngine
            server_args = vLLMConfig.build_args(
                vllm_config=self.config.vllm,
                tp_size=self.allocation_mode.gen.tp_size,
                pp_size=self.allocation_mode.gen.pp_size,
            )
        else:
            raise ValueError(
                f"Invalid backend: {self.allocation_mode.gen_backend}, "
                "expected sglang or vllm"
            )

        controller = engine_cls.as_controller(config, self.scheduler)
        controller.initialize(
            role="eval-rollout",
            alloc_mode=self.allocation_mode,
            server_args=server_args,
        )
        return controller

    def _validate_cfg(self) -> None:
        if (
            self.allocation_mode.gen_backend == "vllm"
            and self.config.rollout.return_routed_experts
        ):
            raise ValueError(
                "return_routed_experts is only supported with SGLang backend. "
                "Please disable return_routed_experts or switch to SGLang backend."
            )

    def _validate_eval_allocation_mode(self) -> None:
        if len(self.allocation_mode.allocations) != 1:
            raise ValueError(
                "RLEvaluator requires inference-only allocation_mode with exactly "
                f"one allocation. Got {self.config.allocation_mode!r}; use a single "
                'server allocation such as "sglang:d4p1t1" for standalone eval.'
            )

        allocation = self.allocation_mode.allocations[0]
        if allocation.backend not in ("sglang", "vllm"):
            raise ValueError(
                "RLEvaluator requires an inference backend in allocation_mode, "
                'for example "sglang:d4p1t1" or "vllm:d4p1t1".'
            )
        if self.allocation_mode.type_ != AllocationType.LLM_SERVER_ONLY:
            raise ValueError(
                "RLEvaluator requires inference-only allocation_mode. "
                f"Got {self.config.allocation_mode!r}; use a single server "
                'allocation such as "sglang:d4p1t1" for standalone eval.'
            )

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs: dict[str, Any] | dict[str, dict[str, Any]],
    ) -> None:
        if isinstance(self.valid_dataloader, dict):
            for dataset_name, dataloader in self.valid_dataloader.items():
                workflow_kwargs = eval_workflow_kwargs[dataset_name]
                cnt = self._submit_dataloader(
                    dataloader=dataloader,
                    eval_workflow=eval_workflow,
                    eval_workflow_kwargs=workflow_kwargs,
                    dataset_name=dataset_name,
                )
                self.eval_rollout.wait(cnt, timeout=None)
            return

        cnt = self._submit_dataloader(
            dataloader=self.valid_dataloader,
            eval_workflow=eval_workflow,
            eval_workflow_kwargs=eval_workflow_kwargs,
            dataset_name=None,
        )
        self.eval_rollout.wait(cnt, timeout=None)

    def _submit_dataloader(
        self,
        dataloader: StatefulDataLoader,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs: dict[str, Any],
        dataset_name: str | None,
    ) -> int:
        cnt = 0
        for data in dataloader:
            for item in data:
                self.eval_rollout.submit(
                    item,
                    eval_workflow,
                    eval_workflow_kwargs,
                    group_size=self.config.eval_gconfig.n_samples,
                    is_eval=True,
                )
                cnt += 1
        if dataset_name is None:
            logger.info("Submitted %d samples evaluated.", cnt)
        else:
            logger.info("Submitted %d samples evaluated, dataset: '%s'.", cnt, dataset_name)
        return cnt

    def _save_log_metrics(
        self, metrics: dict[str, float], epoch: int, epoch_step: int, global_step: int
    ) -> None:
        while True:
            try:
                save_path = os.path.join(
                    self.config.cluster.fileroot,
                    "logs/root",
                    self.config.experiment_name,
                    self.config.trial_name,
                    f"{self.config.experiment_name}_{self.config.trial_name}"
                    "_logger_metrics.jsonl",
                )
                logger.info("Saving metrics to %s", save_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, mode="a+", encoding="utf-8") as f:
                    record = dict(metrics)
                    record["epoch"] = epoch
                    record["epoch_step"] = epoch_step
                    record["global_step"] = global_step
                    f.write(json.dumps(record) + "\n")
                break
            except Exception:
                traceback.print_exc()
                sleep(5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        self.close()
        if exc_type is not None:
            raise exc_value
