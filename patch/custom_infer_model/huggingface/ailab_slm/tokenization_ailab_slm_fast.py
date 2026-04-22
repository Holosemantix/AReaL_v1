# coding=utf-8
## ## adapted from huggingface llama
"""Fast Tokenization classes for AILabSLM model"""

import os
from shutil import copyfile
from typing import Optional, Tuple

from tokenizers import processors

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import is_sentencepiece_available, logging

from tokenizers import decoders, normalizers, pre_tokenizers
from transformers.convert_slow_tokenizer import SpmConverter, _get_prepend_scheme

if is_sentencepiece_available():
    from .tokenization_ailab_slm import AILabSLMTokenizer
else:
    AILabSLMTokenizer = None

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model", "tokenizer_file": "tokenizer.json"}


class AILabSLMConverter(SpmConverter):
    """
    customized AILabSLMConverter to convert AILabSLMTokenizer to AILabSLMTokenizerFast
    """
    handle_byte_fallback = True

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def normalizer(self, proto):
        precompiled_charsmap = proto.normalizer_spec.precompiled_charsmap
        _normalizers = [normalizers.NFKC(), ]
        if precompiled_charsmap:
            return normalizers.Sequence(_normalizers + [normalizers.Precompiled(precompiled_charsmap)])
        return normalizers.Sequence(_normalizers)

    def pre_tokenizer(self, replacement, add_prefix_space):
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self.original_tokenizer)
        return pre_tokenizers.Metaspace(replacement=replacement, prepend_scheme=prepend_scheme, split=False)

    def post_processor(self):
        # the processor is defined in the LlamaTokenizerFast class.
        return None


from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
SLOW_TO_FAST_CONVERTERS = SLOW_TO_FAST_CONVERTERS.update({"AILabSLMTokenizer": AILabSLMConverter})


class AILabSLMTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a AILabSLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and no normalization.

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<pad>"`)
        cls_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<cls>"`)
        mask_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<mask>"`)
        sep_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<sep>"`)
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not the tokenizer should automatically add a prefix space
    """

    vocab_files_names = VOCAB_FILES_NAMES
    slow_tokenizer_class = AILabSLMTokenizer
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        sep_token="<sep>",
        add_bos_token=False,
        add_eos_token=False,
        add_prefix_space=False,
        **kwargs,
    ):

        if add_prefix_space is not None:
            kwargs["from_slow"] = True

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sep_token=sep_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.vocab_file = vocab_file

    @property
    def can_save_slow_tokenizer(self) -> bool:
        return os.path.isfile(self.vocab_file) if self.vocab_file else False

    def update_post_processor(self):
        """
        - functioning of `add_bos_token` and `add_eos_token`
        - Copied from transformers.models.llama.tokenization_llama.LlamaTokenizerFast

        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        single = f"{(bos + ':0 ') if self.add_bos_token else ''}$A:0{(' ' + eos + ':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' ' + bos + ':1') if self.add_bos_token else ''} $B:1{(' ' + eos + ':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not self.can_save_slow_tokenizer:
            raise ValueError(
                "Your fast tokenizer does not have the necessary information to save the vocabulary for a slow "
                "tokenizer."
            )

        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    # Copied from transformers.models.llama.tokenization_llama.LlamaTokenizer.build_inputs_with_special_tokens
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output


__all__ = ["AILabSLMTokenizerFast"]

