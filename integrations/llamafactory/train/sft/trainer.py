# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from webkd.training.token_weights import (
    kl_gamma_enabled,
    applied_weight_sum,
    normalization_mode,
    token_weights_enabled,
    valid_token_count,
    weighted_lm_loss,
)
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")

        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_old_version, self.accelerator
            )
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False)
                    or getattr(ref_model, "is_loaded_in_4bit", False)
                ):
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: (
                eaft_loss_func(outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha)
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        self._token_weights = token_weights_enabled()
        if self._token_weights:
            if self.compute_loss_func is not None or self.label_smoother is not None:
                raise ValueError(
                    "WEBKD_TOKEN_WEIGHTS cannot be combined with dft/eaft/asft loss or label smoothing"
                )

            if self.args.n_gpu > 1:
                raise ValueError(
                    "WEBKD_TOKEN_WEIGHTS does not support DataParallel (n_gpu > 1); use torchrun/DDP"
                )

            self.args.remove_unused_columns = False

            self._webkd_norm = normalization_mode()
            if self._webkd_norm == "window":
                if not hasattr(Seq2SeqTrainer, "_get_num_items_in_batch"):
                    raise RuntimeError(
                        "this transformers version has no Trainer._get_num_items_in_batch; "
                        f"WEBKD_TOKEN_WEIGHT_NORM={self._webkd_norm} cannot guarantee its normaliser here"
                    )

                self.model_accepts_loss_kwargs = True
            else:
                self.model_accepts_loss_kwargs = False

            logger.info_rank0(
                f"WebKD token weights ON: {os.environ.get('WEBKD_TOKEN_WEIGHTS')} (norm={self._webkd_norm})"
            )

        self._webkd_kl_beta = float(os.environ.get("WEBKD_KL_BETA", "0") or 0.0)
        self._webkd_kl_as_gamma = os.environ.get("WEBKD_KL_AS_GAMMA", "0") == "1"
        self._webkd_kl_ref_path = os.environ.get("WEBKD_KL_REF", "")
        self._webkd_kl_ref = None
        if self._webkd_kl_beta > 0.0:
            if not self._webkd_kl_ref_path:
                raise ValueError(
                    "WEBKD_KL_BETA is set but WEBKD_KL_REF (the base-policy checkpoint) is not"
                )
            if self.args.n_gpu > 1:
                raise ValueError("WEBKD_KL_BETA does not support DataParallel (n_gpu > 1)")

            self._webkd_kl_gamma_col = kl_gamma_enabled()
            if self._token_weights and not self._webkd_kl_as_gamma and not self._webkd_kl_gamma_col:
                raise ValueError(
                    "WEBKD_KL_BETA with WEBKD_TOKEN_WEIGHTS needs either WEBKD_KL_AS_GAMMA=1 (weights ARE the "
                    "anchor mask, uniform CE) or WEBKD_KL_GAMMA_WEIGHTS=<side-car> (separate mask, "
                    "weighted CE); otherwise one vector would serve both roles"
                )
            if self._webkd_kl_as_gamma and self._webkd_kl_gamma_col:
                raise ValueError("set WEBKD_KL_AS_GAMMA=1 or WEBKD_KL_GAMMA_WEIGHTS, not both")
            self.args.remove_unused_columns = False
            self.model_accepts_loss_kwargs = False
            logger.info_rank0(
                f"WebKD KL anchor ON: beta={self._webkd_kl_beta} ref={self._webkd_kl_ref_path} "
                f"gamma={'token_weights-as-gamma' if self._webkd_kl_as_gamma else (os.environ.get('WEBKD_KL_GAMMA_WEIGHTS') or 'uniform')}"
                f" ce={'weighted' if (self._token_weights and not self._webkd_kl_as_gamma) else 'uniform'}"
            )

        if training_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self, *args, **kwargs) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer(*args, **kwargs)

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _webkd_weighted_compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        """WebKD weighted cross-entropy: ``sum_i w_i * CE_i / sum_i w_i``."""
        inputs = dict(inputs)
        weights = inputs.pop("token_weights", None)
        labels = inputs.pop("labels", None)
        if weights is None or labels is None:
            raise ValueError(
                "WEBKD_TOKEN_WEIGHTS is on but this batch carries no token_weights/labels"
            )

        mode = getattr(self, "_webkd_norm", "batch")
        if mode == "batch":
            num_items_in_batch = None
        elif mode == "count":
            num_items_in_batch = valid_token_count(labels)

        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        loss = weighted_lm_loss(logits, labels, weights, normalizer=num_items_in_batch)
        if (
            mode == "window"
            and self.args.average_tokens_across_devices
            and num_items_in_batch is not None
        ):
            loss = loss * self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    def _get_webkd_kl_ref(self, device):
        """Frozen base policy, loaded once and kept in eval/bf16 with grads off."""
        if self._webkd_kl_ref is None:
            from transformers import AutoModelForCausalLM
            from transformers.integrations import deepspeed as hf_ds

            prev = getattr(hf_ds, "_hf_deepspeed_config_weak_ref", None)
            hf_ds._hf_deepspeed_config_weak_ref = None
            try:
                logger.info_rank0(f"loading KL reference policy from {self._webkd_kl_ref_path}")

                ref = AutoModelForCausalLM.from_pretrained(
                    self._webkd_kl_ref_path,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map={"": device},
                )
            finally:
                hf_ds._hf_deepspeed_config_weak_ref = prev

            ref.eval().requires_grad_(False)
            self._webkd_kl_ref = ref.to(device)
            p = next(ref.parameters())
            if p.dim() < 2 and p.numel() > 1:
                raise RuntimeError(
                    "KL reference model came back sharded; zero.Init was not suspended"
                )
        return self._webkd_kl_ref

    @staticmethod
    def _webkd_target_logprobs(logits, targets, mask):
        """log p(target_i) at each supervised position, via fused CE (never a full log_softmax)."""
        safe = targets.masked_fill(~mask, 0)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), safe.reshape(-1), reduction="none"
        ).view_as(safe)
        return -nll

    def _webkd_kl_compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        """Uniform SFT cross-entropy plus a role-masked anchor to the base policy."""
        inputs = dict(inputs)
        wcol = inputs.pop("token_weights", None)
        gcol = inputs.pop("kl_gamma", None)
        labels = inputs.pop("labels", None)
        if labels is None:
            raise ValueError("WEBKD_KL_BETA is on but this batch carries no labels")
        if self._webkd_kl_as_gamma and wcol is None:
            raise ValueError("WEBKD_KL_AS_GAMMA=1 but this batch carries no token_weights column")
        if self._webkd_kl_gamma_col and gcol is None:
            raise ValueError(
                "WEBKD_KL_GAMMA_WEIGHTS is set but this batch carries no kl_gamma column"
            )

        gamma = wcol if self._webkd_kl_as_gamma else gcol
        ce_w = None if self._webkd_kl_as_gamma else wcol

        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        mask = shift_labels.ne(-100)

        logp = self._webkd_target_logprobs(shift_logits, shift_labels, mask)
        n = mask.sum().clamp(min=1)
        if ce_w is None:
            ce = -(logp * mask).sum() / n
        else:
            w = ce_w[..., 1:].float() * mask.float()
            ce = -(logp * w).sum() / w.sum().clamp(min=1e-6)

        with torch.no_grad():
            ref = self._get_webkd_kl_ref(logits.device)
            ref_out = ref(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask", None)
            )
            ref_logits = (ref_out["logits"] if isinstance(ref_out, dict) else ref_out[0])[
                ..., :-1, :
            ]
            ref_logp = self._webkd_target_logprobs(ref_logits, shift_labels, mask)

        d = (ref_logp - logp).clamp(-20.0, 8.0)
        kl = torch.exp(d) - d - 1.0

        g = mask.float() if gamma is None else (gamma[..., 1:].float() * mask.float())
        kl_term = (kl * g).sum() / g.sum().clamp(min=1e-6)
        loss = ce + self._webkd_kl_beta * kl_term

        if self.state.global_step % 10 == 0:
            logger.info_rank0(
                f"[webkd-anchor] step={self.state.global_step} ce={ce.item():.4f} "
                f"kl={kl_term.item():.4f} anchored_frac={(g.sum() / n).item():.3f}"
            )

        return (loss, outputs) if return_outputs else loss

    def _get_num_items_in_batch(self, batch_samples, device):
        """WebKD ``window`` mode: normalise by the accumulation window's WEIGHT SUM."""
        if (
            not getattr(self, "_token_weights", False)
            or getattr(self, "_webkd_norm", "batch") != "window"
        ):
            return super()._get_num_items_in_batch(batch_samples, device)

        if not batch_samples or "token_weights" not in batch_samples[0]:
            raise ValueError(
                "WEBKD_TOKEN_WEIGHTS is on but the batches carry no token_weights column"
            )

        total = sum(
            applied_weight_sum(batch["labels"], batch["token_weights"]) for batch in batch_samples
        )
        total = total.to(device)
        if self.args.average_tokens_across_devices and self.args.world_size > 1:
            total = self.accelerator.gather(total).sum()

        return total

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):

        if getattr(self, "_webkd_kl_beta", 0.0) > 0.0:
            return self._webkd_kl_compute_loss(model, inputs, *args, **kwargs)

        if getattr(self, "_token_weights", False):
            return self._webkd_weighted_compute_loss(model, inputs, *args, **kwargs)

        if self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Remove the prompt part in the generated tokens."""
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
            inputs.pop("token_weights", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self,
        dataset: "Dataset",
        predict_results: "PredictionOutput",
        skip_special_tokens: bool = True,
    ) -> None:
        """Save model predictions to `output_dir`."""
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX,
            predict_results.label_ids,
            self.processing_class.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(
            input_ids_list, skip_special_tokens=False
        )
        decoded_preds = self.processing_class.batch_decode(
            preds, skip_special_tokens=skip_special_tokens
        )
        decoded_labels = self.processing_class.batch_decode(
            labels, skip_special_tokens=skip_special_tokens
        )

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(
                    json.dumps(
                        {"prompt": text, "predict": pred, "label": label}, ensure_ascii=False
                    )
                    + "\n"
                )
