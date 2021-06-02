# coding=utf-8
# Copyright 2020 The Fairseq Authors and The HuggingFace Inc. team.
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
""" BART configuration """

from transformers import BartConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

DIALOGBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/bart-base": "https://huggingface.co/facebook/bart-base/resolve/main/config.json",
    "facebook/bart-large": "https://huggingface.co/facebook/bart-large/resolve/main/config.json",
    "facebook/bart-large-mnli": "https://huggingface.co/facebook/bart-large-mnli/resolve/main/config.json",
    "facebook/bart-large-cnn": "https://huggingface.co/facebook/bart-large-cnn/resolve/main/config.json",
    "facebook/bart-large-xsum": "https://huggingface.co/facebook/bart-large-xsum/resolve/main/config.json",
    "facebook/mbart-large-en-ro": "https://huggingface.co/facebook/mbart-large-en-ro/resolve/main/config.json",
    "yjernite/bart_eli5": "https://huggingface.co/yjernite/bart_eli5/resolve/main/config.json",
}


class DialogBartConfig(BartConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.DialogBartModel`. It is used to
    instantiate a DialogBART model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.DialogRobertaConfig` class directly inherits :class:`~transformers.BartConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Args:
        turn_idx: (:obj:`int`, `optional`, defaults to 1721):
            the tolen_id of the token seprating the conversation turns.  tokenizer.convert_tokens_to_ids('Ġ|') is 1721, that is ' |' token
        max_turn_embeddings: (:obj:`int`, `optional`, defaults to 150):
            The maximum number of turns this model might ever be used with.
        max_speaker_embeddings: (:obj:`int`, `optional`, defaults to 10):
            The maximum number of different speakers this model might ever be used with.
        speaker_ids: the token ids of ['ĠAgent', 'ĠCustomer'] in BartTokenizer
    """
    model_type = "dialogbart"
    def __init__(
        self,
        turn_token_id=1721,
        max_turn_embeddings=150,
        max_speaker_embeddings=10,
        speaker_ids = [18497,19458],
        hierarchical_layers = 1,
        **kwargs
    ):
        r"""
        :class:`~transformers.BartConfig` is the configuration class for `BartModel`.
    
        Examples::

            >>> from dialogbart import BartConfig, DialogBartModel
            >>> config = BartConfig.from_pretrained('facebook/bart-large')
            >>> model = BartModel(config)

        """
        if "hidden_size" in kwargs:
            raise ValueError("hidden size is called d_model")
        super().__init__(turn_token_id=turn_token_id, 
        max_turn_embeddings=max_turn_embeddings, 
        max_speaker_embeddings=max_speaker_embeddings,
        speaker_ids = speaker_ids,
        **kwargs )