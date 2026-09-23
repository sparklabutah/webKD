# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, greedy_knapsack, infer_seqlen
from webkd.training.surgical_labels import mask_target_label, surgical_enabled
from webkd.training.token_weights import (
    kl_gamma_enabled,
    target_kl_gamma,
    target_token_weights,
    token_weights_enabled,
)

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput

logger = logging.get_logger(__name__)

MAX_SU_SEQ_IDX = 2**32


@dataclass
class PackingParams:
    """Metadata for a packed sequence: sub-sequence boundaries and multimodal data indices."""

    sequence_boundaries: list[int]
    image_subseq_ids: list[int]
    video_subseq_ids: list[int]
    audio_subseq_ids: list[int]
    right_padding_length: int


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(
            prompt + response, images, videos, audios, self.processor
        )
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        discarding_history_cot = self.data_args.mask_history and not self.template.preserve_thinking
        encoded_pairs = self.template.encode_multiturn(
            self.tokenizer, messages, system, tools, discarding_history_cot
        )
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]

        weighted = token_weights_enabled()
        token_weights = [0.0] * len(input_ids) if weighted else None
        self._token_weights = None

        gammaed = kl_gamma_enabled()
        kl_gamma = [0.0] * len(input_ids) if gammaed else None
        self._webkd_kl_gamma = None

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if surgical_enabled():
                if self.data_args.mask_history:
                    raise ValueError(
                        "WEBKD_SURGICAL_MASK is incompatible with mask_history (turn order is reversed)"
                    )

                target_label = mask_target_label(
                    self.tokenizer, messages[2 * turn_idx + 1]["content"], target_ids, target_label
                )

            target_weight = None
            if weighted:
                if self.data_args.mask_history or self.data_args.packing:
                    raise ValueError(
                        "WEBKD_TOKEN_WEIGHTS is incompatible with mask_history / packing"
                    )

                if surgical_enabled():
                    raise ValueError(
                        "set WEBKD_TOKEN_WEIGHTS or WEBKD_SURGICAL_MASK, not both -- a weight side-car can carry "
                        "hard 'mask_spans' itself, so the two would fight over the same labels"
                    )

                target_weight, target_label = target_token_weights(
                    self.tokenizer, messages[2 * turn_idx + 1]["content"], target_ids, target_label
                )

            if gammaed:
                target_gamma = target_kl_gamma(
                    self.tokenizer, messages[2 * turn_idx + 1]["content"], target_ids, target_label
                )

            if self.data_args.mask_history:
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
                if weighted:
                    token_weights += [0.0 if lab == IGNORE_INDEX else 1.0 for lab in source_label]
                    token_weights += target_weight

                if gammaed:
                    kl_gamma += [0.0 if lab == IGNORE_INDEX else 1.0 for lab in source_label]
                    kl_gamma += target_gamma

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]
            if weighted:
                token_weights += [1.0]

            if gammaed:
                kl_gamma += [1.0]

        if weighted:
            if len(token_weights) != len(labels):
                raise ValueError("WebKD token weights are not aligned with the labels")

            self._token_weights = token_weights

        if gammaed:
            if len(kl_gamma) != len(labels):
                raise ValueError("WebKD KL gamma is not aligned with the labels")

            self._webkd_kl_gamma = kl_gamma

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:

        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(
                        examples["_prompt"][i] + examples["_response"][i]
                    )
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            if getattr(self, "_token_weights", None) is not None:
                model_inputs["token_weights"].append(self._token_weights)

            if getattr(self, "_webkd_kl_gamma", None) is not None:
                model_inputs["kl_gamma"].append(self._webkd_kl_gamma)

            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print(
            "inputs:\n{}".format(
                self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)
            )
        )
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:

        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(
                        examples["_prompt"][i] + examples["_response"][i]
                    )
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(
                    f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}."
                )
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        model_inputs = defaultdict(list)
        requires_packing_params = self.data_args.neat_packing
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = (
                [],
                [],
                [],
                [],
            )
            packed_images, packed_videos, packed_audios = [], [], []
            if requires_packing_params:
                sequence_boundaries = [0]
                image_subseq_ids: list[int] = []
                video_subseq_ids: list[int] = []
                audio_subseq_ids: list[int] = []

            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                if requires_packing_params:
                    n_img = len(batch_images[index])
                    n_vid = len(batch_videos[index])
                    n_aud = len(batch_audios[index])
                    sequence_boundaries.append(
                        sequence_boundaries[-1] + len(batch_input_ids[index])
                    )
                    image_subseq_ids.extend([i] * n_img)
                    video_subseq_ids.extend([i] * n_vid)
                    audio_subseq_ids.extend([i] * n_aud)

                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            if len(packed_input_ids) < self.data_args.cutoff_len + 1:
                pad_length = self.data_args.cutoff_len - len(packed_input_ids) + 1
                packed_input_ids += [self.tokenizer.pad_token_id] * pad_length
                packed_position_ids += [0] * pad_length
                packed_labels += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_attention_masks += [0] * pad_length
                else:
                    packed_attention_masks += [1] * pad_length

                if requires_packing_params:
                    sequence_boundaries.append(sequence_boundaries[-1] + pad_length)

            if len(packed_input_ids) != self.data_args.cutoff_len + 1:
                raise ValueError(
                    "The length of packed example should be identical to the cutoff length."
                )

            model_inputs["input_ids"].append(packed_input_ids)
            if requires_packing_params:
                packing_params = PackingParams(
                    sequence_boundaries=sequence_boundaries,
                    image_subseq_ids=image_subseq_ids or [MAX_SU_SEQ_IDX],
                    video_subseq_ids=video_subseq_ids or [MAX_SU_SEQ_IDX],
                    audio_subseq_ids=audio_subseq_ids or [MAX_SU_SEQ_IDX],
                    right_padding_length=pad_length,
                )
                model_inputs["packing_params"].append(asdict(packing_params))

            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)

        return model_inputs
