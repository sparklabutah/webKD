from typing import Callable, Optional, Union, Any

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
import numpy as np
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available, is_rich_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.utils import pad, nanmin, nanmax

from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

import inspect
import os


def fresh_episode_advantages(rewards_per_row, episode_id, task_index, scale_impl="group", eps=1e-4):
    """Episode-grouped GRPO advantage for fresh-rollout mode (single-turn-like-SFT)."""
    import numpy as _np

    r = _np.asarray(rewards_per_row, dtype=float)
    ep = _np.asarray(episode_id)
    tk = _np.asarray(task_index)
    n = len(r)
    if n == 0:
        return [], []

    ep_ids, first_idx = _np.unique(ep, return_index=True)
    ep_reward = {int(e): float(r[i]) for e, i in zip(ep_ids, first_idx)}
    ep_task = {int(e): int(tk[i]) for e, i in zip(ep_ids, first_idx)}

    task_to_eps = {}
    for e in ep_ids:
        task_to_eps.setdefault(ep_task[int(e)], []).append(int(e))
    ep_adv, ep_std = {}, {}
    all_ep_r = _np.asarray([ep_reward[int(e)] for e in ep_ids], dtype=float)
    batch_std = float(all_ep_r.std()) if len(all_ep_r) > 1 else 0.0
    for task, eps_list in task_to_eps.items():
        vals = _np.asarray([ep_reward[e] for e in eps_list], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std()) if len(vals) > 1 else 0.0
        for e in eps_list:
            a = ep_reward[e] - mean
            if scale_impl in ("group", "true", "1"):
                a = a / (std + eps)
            elif scale_impl == "batch":
                a = a / (batch_std + eps)
            elif scale_impl in ("none", "false", "0", "off"):
                pass
            else:
                raise ValueError(f"Unknown scale_impl={scale_impl!r}")
            ep_adv[e] = a
            ep_std[e] = std
    return [ep_adv[int(e)] for e in ep], [ep_std[int(e)] for e in ep]


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOEnvTrainer(GRPOTrainer):
    """WebAgent-R1 mGRPO trainer, re-ported to trl 0.24 **vLLM server mode**."""

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        env: Environment,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        **kwargs,
    ):
        if not args.use_vllm:
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if getattr(args, "vllm_mode", "server") != "server":
            raise ValueError("GRPOEnvTrainer (server re-port) requires args.vllm_mode == 'server'")
        if not (
            callable(reward_funcs)
            or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))
        ):
            raise ValueError("reward_funcs must be a function or a list of functions.")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        self.env = env

        _lt = os.environ.get("TW_LOSS_TYPE", "").strip().lower()
        self.tw_loss_type = _lt if _lt in ("turn", "turn_const", "mgrpo") else None

        self.tw_is_level = os.environ.get("TW_IS_LEVEL", "").strip().lower() or None

        self.tw_turn_decay = float(os.environ.get("TW_TURN_DECAY", "1.0"))

        self.tw_turn_tmax = int(os.environ.get("TW_TURN_TMAX", str(getattr(env, "max_steps", 10))))

        self.tw_turn_p = float(os.environ.get("TW_TURN_P", "1.0"))
        self.tw_turn_floor = float(os.environ.get("TW_TURN_NORM_FLOOR", "0"))
        assert 0.0 <= self.tw_turn_p <= 1.0, "TW_TURN_P must be in [0, 1]"

        _bt = os.environ.get("TW_BETA_THINK")
        _ba = os.environ.get("TW_BETA_ACTION")

        _bp = os.environ.get("TW_BETA_PLAN")
        _bm = os.environ.get("TW_BETA_MEMORY")
        if _bt is not None or _ba is not None or _bp is not None or _bm is not None:
            _default_beta = self.beta if self.beta else 5e-4
            self.tw_beta_think = float(_bt) if _bt is not None else _default_beta
            self.tw_beta_action = float(_ba) if _ba is not None else _default_beta
            self.tw_beta_plan = float(_bp) if _bp is not None else self.tw_beta_think
            self.tw_beta_memory = float(_bm) if _bm is not None else self.tw_beta_think

            if self.beta == 0.0:
                raise ValueError(
                    "TW_BETA_THINK/TW_BETA_ACTION set but args.beta == 0 — the reference-model "
                    "forward is disabled, so per-role KL cannot be computed. Set beta != 0 "
                    "(train_timewarp.py does this automatically)."
                )
        else:
            self.tw_beta_think = None
            self.tw_beta_action = None
            self.tw_beta_plan = None
            self.tw_beta_memory = None

        self._tag_active = (
            self.tw_loss_type is not None
            or self.tw_is_level == "turn"
            or self.tw_turn_decay != 1.0
            or self.tw_beta_think is not None
        )
        if self._tag_active and self.accelerator.is_main_process:
            print(
                ">>> [TAG] active: "
                f"loss_type={self.tw_loss_type} is_level={self.tw_is_level} "
                f"turn_decay={self.tw_turn_decay} T_max={self.tw_turn_tmax} "
                f"turn_p={self.tw_turn_p} turn_floor={self.tw_turn_floor} "
                f"beta_think={self.tw_beta_think} beta_action={self.tw_beta_action}",
                flush=True,
            )

    def _logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size):
        fn = getattr(self, "_get_per_token_logps_and_entropies", None)
        if fn is None:
            return self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        kw = {}
        params = inspect.signature(fn).parameters
        if "batch_size" in params:
            kw["batch_size"] = batch_size
        out = fn(model, input_ids, attention_mask, logits_to_keep, **kw)
        return out[0] if isinstance(out, tuple) else out

    def _fix_param_name_to_vllm(self, name, extra_prefixes=None):

        name = super()._fix_param_name_to_vllm(name, extra_prefixes=extra_prefixes)
        if name.startswith("model.") or name.startswith("lm_head."):
            name = "language_model." + name
        return name

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = inputs

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        all_prompts = gather_object(prompts)
        all_inputs = gather_object(inputs)

        is_fresh = False
        fresh_prompt_ids = fresh_episode_id = fresh_task_index = None
        if self.accelerator.is_main_process:
            sampling_kwargs = dict(
                temperature=getattr(self, "temperature", 1.0),
                top_p=getattr(self, "top_p", 1.0),
                top_k=(
                    getattr(self, "top_k", None) if getattr(self, "top_k", None) is not None else -1
                ),
                min_p=(
                    getattr(self, "min_p", None)
                    if getattr(self, "min_p", None) is not None
                    else 0.0
                ),
                repetition_penalty=getattr(self, "repetition_penalty", 1.0),
            )
            max_seq_len = int(os.environ.get("TW_MAX_SEQ_LEN", "40960"))
            env_result = self.env.generate(
                prompts=all_prompts,
                task_configs=all_inputs,
                vllm_client=self.vllm_client,
                tokenizer=self.processing_class,
                sampling_kwargs=sampling_kwargs,
                max_seq_len=max_seq_len,
            )
            completion_ids = env_result["ids"]
            completion_mask = env_result["mask"]
            completion_messages = env_result["messages"]
            completion_reward = env_result["reward"]
            completion_steps = env_result["steps"]

            completion_turn_ids = env_result["turn_ids"]
            completion_role_ids = env_result["role_ids"]

            is_fresh = bool(env_result.get("fresh", False))
            if is_fresh:
                if self.accelerator.num_processes != 1:
                    raise RuntimeError(
                        "TW_ROLLOUT_MODE=fresh requires a single train process (num_processes==1)"
                    )
                if os.environ.get("TW_DYNAMIC_SAMPLING", "0").lower() in ("1", "true", "yes", "on"):
                    raise RuntimeError(
                        "TW_DYNAMIC_SAMPLING is unsupported with TW_ROLLOUT_MODE=fresh"
                    )
                fresh_prompt_ids = env_result["prompt_ids"]
                fresh_episode_id = env_result["episode_id"]
                fresh_task_index = env_result["task_index"]

            if os.environ.get("TW_DYNAMIC_SAMPLING", "0").lower() in ("1", "true", "yes", "on"):
                G = self.num_generations
                n_groups = len(completion_reward) // G
                ds_rounds = int(os.environ.get("TW_DS_MAX_ROUNDS", "3"))
                ds_force = os.environ.get("TW_DS_FORCE", "0").lower() in ("1", "true", "yes", "on")
                ds_temp = float(os.environ.get("TW_DS_TEMPERATURE", "1.3"))
                ds_resampled = 0
                for g in range(n_groups):
                    lo, hi = g * G, (g + 1) * G
                    if (float(np.std(completion_reward[lo:hi])) != 0.0) and not ds_force:
                        continue
                    ds_sk = dict(sampling_kwargs)
                    ds_sk["temperature"] = ds_temp
                    for _r in range(ds_rounds):
                        try:
                            rr = self.env.generate(
                                prompts=all_prompts[lo:hi],
                                task_configs=all_inputs[lo:hi],
                                vllm_client=self.vllm_client,
                                tokenizer=self.processing_class,
                                sampling_kwargs=ds_sk,
                                max_seq_len=max_seq_len,
                            )
                        except Exception as _e:
                            print(f">>> [dyn-sample] group {g} re-roll error: {_e}", flush=True)
                            break
                        if float(np.std(rr["reward"])) > 0.0 or (ds_force and _r == ds_rounds - 1):
                            completion_ids[lo:hi] = rr["ids"]
                            completion_mask[lo:hi] = rr["mask"]
                            completion_messages[lo:hi] = rr["messages"]
                            completion_reward[lo:hi] = rr["reward"]
                            completion_steps[lo:hi] = rr["steps"]
                            completion_turn_ids[lo:hi] = rr["turn_ids"]
                            completion_role_ids[lo:hi] = rr["role_ids"]
                            ds_resampled += 1
                            break
                print(
                    f">>> [dyn-sample] resampled {ds_resampled}/{n_groups} degenerate "
                    f"groups (force={ds_force}, rounds<={ds_rounds}, temp={ds_temp})",
                    flush=True,
                )

            if self.max_prompt_length is not None:
                completion_ids = [row[-self.max_prompt_length :] for row in completion_ids]
                completion_mask = [row[-self.max_prompt_length :] for row in completion_mask]

                completion_turn_ids = [
                    row[-self.max_prompt_length :] for row in completion_turn_ids
                ]
                completion_role_ids = [
                    row[-self.max_prompt_length :] for row in completion_role_ids
                ]

            self._pending_tag_tele = {}
            try:
                G = self.num_generations
                _rw = np.asarray(completion_reward, dtype=float)
                if (not is_fresh) and _rw.size >= G and _rw.size % G == 0:
                    _grp_std = _rw.reshape(-1, G).std(axis=1)
                    self._pending_tag_tele["degenerate_group_frac"] = float(
                        (_grp_std == 0.0).mean()
                    )
                _turn_lens = []
                for _tids_row, _mask_row in zip(completion_turn_ids, completion_mask):
                    _per = {}
                    for _tid, _mk in zip(_tids_row, _mask_row):
                        if _tid > 0 and _mk:
                            _per[_tid] = _per.get(_tid, 0) + 1
                    _turn_lens.extend(_per.values())
                if _turn_lens:
                    _tl = np.asarray(_turn_lens, dtype=float)
                    self._pending_tag_tele["turn_len_mean"] = float(_tl.mean())
                    self._pending_tag_tele["turn_len_p90"] = float(np.percentile(_tl, 90))
                    self._pending_tag_tele["turn_len_max"] = float(_tl.max())
                    self._pending_tag_tele["turns_per_traj_mean"] = float(
                        np.mean(
                            [
                                max(r) if (r := [t for t in row if t > 0]) else 0
                                for row in completion_turn_ids
                            ]
                        )
                    )
                _steps = np.asarray(completion_steps, dtype=float)
                if _steps.size:
                    self._pending_tag_tele["turns_max"] = float(_steps.max())
                self._pending_tag_tele["turn_lens_hist"] = list(_turn_lens)
            except Exception as _e:
                print(f">>> [tag-telemetry] skipped: {_e}", flush=True)
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)
            completion_reward = [None] * len(all_prompts)
            completion_steps = [None] * len(all_prompts)
            completion_turn_ids = [None] * len(all_prompts)
            completion_role_ids = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)
        completion_reward = broadcast_object_list(completion_reward, from_process=0)
        completion_steps = broadcast_object_list(completion_steps, from_process=0)
        completion_turn_ids = broadcast_object_list(completion_turn_ids, from_process=0)
        completion_role_ids = broadcast_object_list(completion_role_ids, from_process=0)

        if is_fresh:
            process_slice = slice(0, len(completion_ids))
        else:
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        completion_reward = completion_reward[process_slice]
        completion_steps = completion_steps[process_slice]
        completion_turn_ids = completion_turn_ids[process_slice]
        completion_role_ids = completion_role_ids[process_slice]
        if is_fresh:
            fresh_prompt_ids = fresh_prompt_ids[process_slice]
            fresh_episode_id = fresh_episode_id[process_slice]
            fresh_task_index = fresh_task_index[process_slice]

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        completion_turn_ids = [
            torch.tensor(t, device=device, dtype=torch.long) for t in completion_turn_ids
        ]
        completion_turn_ids = pad(completion_turn_ids, padding_value=0)
        completion_role_ids = [
            torch.tensor(r, device=device, dtype=torch.long) for r in completion_role_ids
        ]
        completion_role_ids = pad(completion_role_ids, padding_value=0)

        if is_fresh:
            _pad_id = self.processing_class.pad_token_id
            _pid = [torch.tensor(p, device=device, dtype=torch.long) for p in fresh_prompt_ids]
            prompt_ids = pad(_pid, padding_value=_pad_id, padding_side="left")
            prompt_mask = (prompt_ids != _pad_id).long()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        if self.max_prompt_length is not None:
            prompt_completion_ids = prompt_completion_ids[:, -self.max_prompt_length :]
            attention_mask = attention_mask[:, -self.max_prompt_length :]
        logits_to_keep = completion_ids.size(1)

        mode = "eval" if self.control.should_evaluate else "train"
        bsz = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            need_old = (self.num_iterations > 1) or (
                self.args.gradient_accumulation_steps % generate_every != 0
            )
            if need_old:
                old_per_token_logps = self._logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, bsz
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, bsz
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, bsz
                    )

        completions = completion_messages
        llm_as_judge_reward = completion_reward

        rewards_per_func = torch.zeros(
            len(completion_reward), len(self.reward_funcs), device=device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
            output_reward_func = reward_func(
                prompts=prompts,
                completions=completions,
                llm_as_judge_reward=llm_as_judge_reward,
                **reward_kwargs,
            )
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        _scale = os.environ.get("TW_SCALE_REWARDS_IMPL", "group").strip().lower()
        if is_fresh:
            _adv, _gstd = fresh_episode_advantages(
                rewards.tolist(),
                fresh_episode_id,
                fresh_task_index,
                scale_impl=_scale,
            )
            advantages = torch.tensor(_adv, dtype=rewards.dtype, device=device)
            std_grouped_rewards = torch.tensor(_gstd, dtype=rewards.dtype, device=device)
            if os.environ.get("TW_FRESH_DEBUG"):
                import collections as _c

                _tc = _c.Counter(int(t) for t in fresh_task_index)
                print(
                    f">>> [fresh-dbg] n_rows={len(_adv)} n_ep={len(set(int(e) for e in fresh_episode_id))} "
                    f"tasks/eps={dict(list(_tc.items())[:6])} "
                    f"reward min/max/mean={rewards.min().item():.4f}/{rewards.max().item():.4f}/{rewards.mean().item():.4f} "
                    f"gstd min/max={min(_gstd):.6f}/{max(_gstd):.6f} "
                    f"ADV min/max/mean={advantages.min().item():.3f}/{advantages.max().item():.3f}/{advantages.mean().item():.3f}",
                    flush=True,
                )
        else:
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

            adv = rewards - mean_grouped_rewards
            if _scale in ("group", "true", "1"):
                adv = adv / (std_grouped_rewards + 1e-4)
            elif _scale == "batch":
                adv = adv / (rewards.std() + 1e-4)
            elif _scale in ("none", "false", "0", "off"):
                pass
            else:
                raise ValueError(f"Unknown TW_SCALE_REWARDS_IMPL={_scale!r} (group|batch|none)")
            advantages = adv

            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            advantages = advantages[process_slice]

        completion_length_t = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        num_items_in_batch = completion_length_t.sum()
        completion_steps_gathered = self.accelerator.gather_for_metrics(completion_steps)
        self._metrics[mode]["completion_length"].append(completion_length_t.float().mean().item())
        self._metrics[mode]["n_rounds"].append(float(np.mean(completion_steps_gathered)))
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            self._metrics[mode][f"rewards/{reward_func.__name__}"].append(reward_per_func[i].item())
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        _tele = getattr(self, "_pending_tag_tele", None)
        if _tele and self.accelerator.is_main_process:
            _hist = _tele.pop("turn_lens_hist", None)
            for _k, _v in _tele.items():
                self._metrics[mode][_k].append(_v)
            if (
                _hist
                and self.args.report_to
                and "wandb" in self.args.report_to
                and is_wandb_available()
                and wandb.run is not None
                and self.state.global_step % self.args.logging_steps == 0
            ):
                try:
                    wandb.log(
                        {
                            f"{mode}/turn_len_hist": wandb.Histogram(_hist),
                            f"{mode}/turns_hist": wandb.Histogram(list(completion_steps_gathered)),
                        },
                        commit=False,
                    )
                except Exception:
                    pass
            self._pending_tag_tele = None

        if (
            self.log_completions
            and not is_fresh
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_log = [x["prompt"] for x in inputs]
            prompts_to_log = gather_object(prompts_log)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()
            if self.accelerator.is_main_process:
                if is_rich_available():
                    idx = rewards.argmax().item()
                    print_prompt_completions_sample(
                        [str(prompts_to_log[idx][-1]["content"])],
                        [completions_to_log[idx]],
                        [rewards_to_log[idx]],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    df = pd.DataFrame(
                        {
                            "step": [str(self.state.global_step)] * len(rewards),
                            "prompt": prompts_to_log,
                            "completion": completions_to_log,
                            "reward": rewards.tolist(),
                        }
                    )
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        out = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "turn_ids": completion_turn_ids,
            "role_ids": completion_role_ids,
        }
        if old_per_token_logps is not None:
            out["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            out["ref_per_token_logps"] = ref_per_token_logps
        return out

    def _compute_loss(self, model, inputs):
        if not self._tag_active:
            return super()._compute_loss(model, inputs)

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        log_ratio = per_token_logps - old_per_token_logps

        tids = inputs["turn_ids"]
        rids = inputs["role_ids"]
        m = completion_mask.float()
        T_max = self.tw_turn_tmax
        K = int(tids.max().item()) + 1
        counts = torch.zeros(m.size(0), K, device=m.device, dtype=m.dtype)
        counts.scatter_add_(1, tids.long(), m)

        if self.tw_is_level == "turn":
            sum_lr = torch.zeros_like(counts)
            sum_lr.scatter_add_(1, tids.long(), log_ratio * m)
            mean_lr = sum_lr / counts.clamp(min=1.0)
            log_importance_weights = mean_lr.gather(1, tids.long())
        elif self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(
                -1
            ).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.tw_turn_decay != 1.0:
            T_i = tids.max(dim=1, keepdim=True).values.float()
            w_pos = self.tw_turn_decay ** (T_i - tids.float())
            per_token_loss = per_token_loss * torch.where(m > 0, w_pos, torch.ones_like(w_pos))

        if self.tw_beta_think is not None:
            beta_tok = torch.full_like(m, self.tw_beta_think)
            beta_tok = torch.where(rids == 2, torch.full_like(m, self.tw_beta_action), beta_tok)
            beta_tok = torch.where(rids == 3, torch.full_like(m, self.tw_beta_plan), beta_tok)
            beta_tok = torch.where(rids == 4, torch.full_like(m, self.tw_beta_memory), beta_tok)
            per_token_loss = per_token_loss + beta_tok * per_token_kl
        elif self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.tw_loss_type == "turn":
            if self.tw_turn_p == 1.0 and self.tw_turn_floor <= 1.0:
                inv = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
            else:
                C_turn = float(self.env.max_tokens_per_turn)
                eff = counts.clamp(min=max(1.0, self.tw_turn_floor))
                inv = torch.where(
                    counts > 0,
                    eff.pow(-self.tw_turn_p) * (C_turn ** (self.tw_turn_p - 1.0)),
                    torch.zeros_like(counts),
                )
            w = inv.gather(1, tids.long()) * m
            loss = (per_token_loss * w).sum() / (per_token_loss.size(0) * T_max)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.tw_loss_type == "turn_const":
            C_turn = self.env.max_tokens_per_turn
            loss = (per_token_loss * m).sum() / (per_token_loss.size(0) * T_max * C_turn)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.tw_loss_type == "mgrpo":
            inv = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
            w = inv.gather(1, tids.long()) * m
            row_sum = (per_token_loss * w).sum(dim=1)
            T_i = tids.max(dim=1).values.clamp(min=1).float()
            loss = (row_sum / T_i).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
            loss = loss / self.current_gradient_accumulation_steps
            if os.environ.get("TW_FRESH_DEBUG"):
                _mf = completion_mask.float()

                _cm = coef_1 * _mf
                _coef_masked_max = _cm.max().item()
                _lrm = log_ratio.abs() * _mf
                _lr_masked_max = _lrm.max().item()
                print(
                    f">>> [fresh-dbg loss] B={per_token_loss.size(0)} max_comp_len={self.max_completion_length} "
                    f"log_ratio|max|(all/masked)={log_ratio.abs().max().item():.3f}/{_lr_masked_max:.3f} "
                    f"coef_1 max(all/masked)={coef_1.max().item():.3f}/{_coef_masked_max:.3f} "
                    f"adv|max|={advantages.abs().max().item():.3f} ptl|max|={per_token_loss.abs().max().item():.3f} "
                    f"maskedtok={int(completion_mask.sum().item())} loss={loss.item():.4f}",
                    flush=True,
                )
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"

        if self.tw_loss_type == "turn":
            self._metrics[mode]["tag_turn_w_max"].append(
                self.accelerator.gather(w.max()).max().item()
            )
            eff_scale = w.sum() / m.sum().clamp(min=1.0)
            self._metrics[mode]["tag_turn_eff_scale"].append(
                self.accelerator.gather(eff_scale).nanmean().item()
            )

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

            for _name, _rid in (("think", 1), ("action", 2), ("plan", 3), ("memory", 4)):
                sel = (rids == _rid).float() * m
                n_sel = sel.sum()
                if _rid in (3, 4) and n_sel.item() == 0:
                    continue
                kl_r = (per_token_kl * sel).sum() / n_sel.clamp(min=1.0)
                self._metrics[mode][f"kl_{_name}"].append(
                    self.accelerator.gather(kl_r).nanmean().item()
                )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss
