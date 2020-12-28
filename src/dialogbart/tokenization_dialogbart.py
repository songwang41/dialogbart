# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

from transformers import add_start_docstrings

from transformers.utils import logging
from transformers import BartTokenizer


logger = logging.get_logger(__name__)


# vocab and merges same as roberta
vocab_url = "https://huggingface.co/roberta-large/resolve/main/vocab.json"
merges_url = "https://huggingface.co/roberta-large/resolve/main/merges.txt"
_all_bart_models = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "yjernite/bart_eli5",
    # This is not exhaustive: see https://huggingface.co/models?filter=bart
]

class DialogBartTokenizer(BartTokenizer):
    r"""
    Construct a DialogBART tokenizer.

    :class:`~transformers.DiglogBartTokenizer` is identical to :class:`~transformers.BartTokenizer`

    Refer to superclass :class:`~transformers.BartTokenizer`, superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning the
    initialization parameters and other methods.
    """
    pass
    
    
