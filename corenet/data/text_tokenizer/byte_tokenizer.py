#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from argparse import Namespace
from typing import List, Union

import torch
from torch import Tensor

from corenet.data.text_tokenizer import TOKENIZER_REGISTRY, BaseTextTokenizer
from corenet.utils import logger
from corenet.utils.ddp_utils import is_rank_0_worker_0

@TOKENIZER_REGISTRY.register(name="byte")
class ByteTokenizer(BaseTextTokenizer):
    """Byte tokenizer that encodes text into a list of UTF-8 byte values.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: Namespace) -> None:
        super().__init__(opts)
    @property
    def sot_token_id(self) -> int:
        return 255

    @property
    def eot_token_id(self) -> int:
        return 254

    @property
    def pad_token_id(self) -> int:
        return 253
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add arguments related to byte tokenizer."""
        if cls == ByteTokenizer:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--text-tokenizer.byte.append_sot_token",
                action="store_true",
                default=False,
                help="Append start of text token before tokenized text. Defaults to False.",
            )
            group.add_argument(
                "--text-tokenizer.byte.append_eot_token",
                action="store_true",
                default=False,
                help="Append end of text token after tokenized text. Defaults to False.",
            )
        return parser

    @property
    def vocab_size(self) -> int:
        """Vocabulary size is fixed to 256 for byte-level encoding."""
        return 256

    def tok_encode(self, input_sentence: str) -> Tensor:
        """Encodes a sentence into a tensor of byte token ids.

        Args:
            input_sentence: Input sentence to be tokenized.

        Returns:
            A tensor containing byte token indices.
        """
        tokenized_seq = list(input_sentence.encode('utf-8'))

        if getattr(self.opts, "text_tokenizer.byte.append_sot_token"):
            tokenized_seq = [self.sot_token_id] + tokenized_seq

        if getattr(self.opts, "text_tokenizer.byte.append_eot_token"):
            tokenized_seq = tokenized_seq + [self.eot_token_id]

        tokenized_seq = torch.tensor(tokenized_seq, dtype=torch.long)
        return tokenized_seq

    def tok_decode(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        """Decodes byte token ids into a sentence.

        Args:
            token_ids: Byte token indices as a list of integers or a 1D integer tensor.

        Returns:
            A decoded sequence.
        """
        if isinstance(token_ids, torch.Tensor):
            assert token_ids.dim() == 1 and token_ids.dtype in [
                torch.int,
                torch.int64,
                torch.int32,
                torch.int8,
            ]
            token_ids = token_ids.numpy().tolist()

        assert isinstance(token_ids, list) and all(
            isinstance(x, int) for x in token_ids
        )
        return bytes(token_ids).decode('utf-8', errors='ignore')
