from typing import TYPE_CHECKING, Optional

from datasets.distributed import split_dataset_by_node

from areal.api.cli_args import _DatasetConfig
from areal.utils import logging

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

VALID_DATASETS = ["gsm8k", "clevr_count_70k", "geometry3k", "hh-rlhf", "torl_data", "bigmath", "ringlite", "math500", "aime24", "aime25", "hmmt25", "nemotron_code"]

logger = logging.getLogger("Dataset")


def _get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    path_list = [p.strip() for p in path.split(',')]
    dataset_map = {}
    for path in path_list:
        if "gsm8k" in path and type == "sft":
            from .gsm8k import get_gsm8k_sft_dataset

            dataset_map['gsm8k'] = get_gsm8k_sft_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "gsm8k" in path and type == "rl":
            from .gsm8k import get_gsm8k_rl_dataset

            dataset_map['gsm8k'] = get_gsm8k_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "clevr_count_70k" in path and type == "sft":
            from .clevr_count_70k import get_clevr_count_70k_sft_dataset

            dataset_map['clevr_count_70k'] = get_clevr_count_70k_sft_dataset(
                path=path,
                split=split,
                processor=processor,
                max_length=max_length,
                **kwargs,
            )
        elif "clevr_count_70k" in path and type == "rl":
            from .clevr_count_70k import get_clevr_count_70k_rl_dataset

            dataset_map['clevr_count_70k'] = get_clevr_count_70k_rl_dataset(
                path=path,
                split=split,
                processor=processor,
                max_length=max_length,
                **kwargs,
            )
        elif "geometry3k" in path and type == "sft":
            from .geometry3k import get_geometry3k_sft_dataset

            dataset_map['geometry3k'] = get_geometry3k_sft_dataset(
                path=path,
                split=split,
                processor=processor,
                max_length=max_length,
                **kwargs,
            )
        elif "geometry3k" in path and type == "rl":
            from .geometry3k import get_geometry3k_rl_dataset

            dataset_map['geometry3k'] = get_geometry3k_rl_dataset(
                path=path,
                split=split,
                processor=processor,
                max_length=max_length,
                **kwargs,
            )
        elif "hh-rlhf" in path and type == "rw":
            from .hhrlhf import get_hhrlhf_rw_dataset

            dataset_map['hh-rlhf'] = get_hhrlhf_rw_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "torl_data" in path and type == "rl":
            from .torl_data import get_torl_data_rl_dataset

            dataset_map['torl_data'] = get_torl_data_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "bigmath" in path.lower().replace('-', '') and type == "rl":
            from .bigmath import get_bigmath_rl_dataset

            dataset_map['bigmath'] = get_bigmath_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "ringlite" in path.lower().replace('-', '') and type == "rl":
            from .ringlite_math import get_ringlite_rl_dataset

            dataset_map['ringlite'] = get_ringlite_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "MATH-500" in path and type == "rl":
            from .math500 import get_math500_rl_dataset

            dataset_map['math500'] = get_math500_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "aime24" in path and type == "rl":
            from .aime24 import get_aime24_rl_dataset

            dataset_map['aime24'] = get_aime24_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "aime25" in path and type == "rl":
            from .aime25 import get_aime25_rl_dataset

            dataset_map['aime25'] = get_aime25_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "aime26" in path and type == "rl":
            from .aime26 import get_aime26_rl_dataset

            dataset_map['aime26'] = get_aime26_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif "hmmt" in path and type == "rl":
            from .hmmt25 import get_hmmt25_rl_dataset

            dataset_map['hmmt25'] = get_hmmt25_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        elif ("nemotron" in path.lower() and "cod" in path.lower()) and type == "rl":
            from .code.nemotron_competitive import get_nemotron_competitive_rl_dataset

            dataset_map['nemotron_code'] = get_nemotron_competitive_rl_dataset(
                path=path,
                split=split,
                tokenizer=tokenizer,
                max_length=max_length,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Dataset {path} with split {split} and training type {type} is not supported. "
                f"Supported datasets are: {VALID_DATASETS}. "
            )

    return dataset_map


def get_custom_dataset_legacy(
    path: str,
    rank: int,
    world_size: int,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    logger.warning(
        "get_custom_dataset using rank and world_size is deprecated. "
        "Please use DistributedSampler in dataloader instead for distributed training."
    )
    dataset = _get_custom_dataset(
        path=path,
        type=type,
        split=split,
        max_length=max_length,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )
    return split_dataset_by_node(dataset, rank=rank, world_size=world_size)


def get_multi_custom_dataset(
    split: str | None = None,
    dataset_config: _DatasetConfig | None = None,
    tokenizer: Optional["PreTrainedTokenizerFast"] = None,
    processor: Optional["ProcessorMixin"] = None,
    **kwargs,
) -> "Dataset":
    if "rank" in kwargs:
        # compatibility for legacy get_custom_dataset
        return get_custom_dataset_legacy(
            split=split,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
        )

    if dataset_config is not None:
        return _get_custom_dataset(
            path=dataset_config.path,
            type=dataset_config.type,
            split=split,
            max_length=dataset_config.max_length,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs,
        )

    # try to pass arguments directly to legacy get_custom_dataset
    logger.warning("dataset_config is not provided")
    return _get_custom_dataset(
        split=split,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )
