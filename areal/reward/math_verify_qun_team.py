from math_verify import parse, verify

from areal.utils import logging

logger = logging.getLogger("MathVerifyReward")


def math_verify_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        parsed_completions = parse(completions)
        parsed_answer = parse("\\boxed{" + str(answer) + "}")
        is_equal = verify(gold=parsed_answer, target=parsed_completions, float_rounding=2, allow_set_relation_comp=True)
        return 1.0 if is_equal else 0.0
    except Exception:
        logger.warning("Exception in math_verify_reward_fn", exc_info=True)
        return 0.0
