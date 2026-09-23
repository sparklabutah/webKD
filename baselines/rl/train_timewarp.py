"""Multi-turn policy-gradient baseline for web agents."""

import os
from transformers import TrainerCallback

if os.environ.get("TW_DIAG") == "1":
    from torch.distributed import distributed_c10d as _c10d

    _orig_pgn = _c10d._process_group_name

    def _logged_pgn(ranks, use_hashed_name):
        before = _c10d._world.group_count
        name = _orig_pgn(ranks, use_hashed_name)
        print(
            f">>> [PGN pid{os.getpid()}] ranks={ranks} hashed={use_hashed_name} "
            f"count{before}->{_c10d._world.group_count} name={name}",
            flush=True,
        )
        return name

    _c10d._process_group_name = _logged_pgn

import verifiers as vf
from verifiers.envs.timewarp_env import TimeWarpEnv

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-4B")
NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
MAX_STEPS = int(os.environ.get("TW_MAX_STEPS", "10"))
SEED = int(os.environ.get("TW_SEED", "42"))
MASK_THINKING = os.environ.get("TW_MASK_THINKING", "0") in ("1", "true", "True")
MAX_OBS_TOKENS = int(os.environ.get("TW_MAX_OBS_TOKENS", "4096"))
MAX_TOKENS_PER_TURN = int(os.environ.get("TW_MAX_TOKENS_PER_TURN", "1024"))
OUTPUT_DIR = os.environ.get("TW_OUTPUT_DIR", os.path.abspath("./outputs"))
RUN_NAME = os.environ.get("TW_RUN_NAME", f"timewarp_GRPO/{MODEL_PATH.split('/')[-1].lower()}")

SITES_FILTER = [s for s in os.environ.get("TW_SITES", "").split(",") if s] or None
TASK_LIMIT = int(os.environ.get("TW_TASK_LIMIT", "-1"))
TRAIN_MAX_STEPS = int(os.environ.get("TW_TRAIN_MAX_STEPS", "-1"))
GRAD_ACCUM = int(os.environ.get("TW_GRAD_ACCUM", "-1"))
REPORT_TO = os.environ.get("TW_REPORT_TO")
SAVE_STRATEGY = os.environ.get("TW_SAVE_STRATEGY")

BETA = os.environ.get("TW_BETA")
NUM_GENERATIONS = int(os.environ.get("TW_NUM_GENERATIONS", "-1"))
PER_DEVICE_BATCH = int(os.environ.get("TW_PER_DEVICE_TRAIN_BATCH_SIZE", "-1"))

for var in ("TW_WIKI", "TW_NEWS", "TW_WEBSHOP", "TW_HOME"):
    assert var in os.environ, f"Set {var} (point it at the running TimeWarp server)"
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY for the TimeWarp LLM judge"

model, tokenizer = vf.get_model_and_tokenizer(MODEL_PATH)

if os.environ.get("TW_GDN_NO_GQA_EXPAND", "0") == "1":
    _patched = 0
    for _m in model.modules():
        _nv = getattr(_m, "num_v_heads", None)
        _nk = getattr(_m, "num_k_heads", None)
        if _nv and _nk and _nv // _nk > 1 and hasattr(_m, "chunk_gated_delta_rule"):

            def _mk(_orig, _r):
                def _wrapped(q, k, v, **kw):

                    return _orig(q[:, :, ::_r].contiguous(), k[:, :, ::_r].contiguous(), v, **kw)

                return _wrapped

            _m.chunk_gated_delta_rule = _mk(_m.chunk_gated_delta_rule, _nv // _nk)
            _patched += 1
    print(
        f">>> TW_GDN_NO_GQA_EXPAND=1: un-expanded GQA on {_patched} GDN layers "
        f"(fla sm_120 misaligned-address workaround; math-identical)"
    )

if not hasattr(model, "warnings_issued") or not isinstance(getattr(model, "warnings_issued"), dict):
    model.warnings_issued = {}

vf_env = TimeWarpEnv(
    max_steps=MAX_STEPS,
    seed=SEED,
    mask_thinking=MASK_THINKING,
    max_obs_tokens=MAX_OBS_TOKENS,
    max_tokens_per_turn=MAX_TOKENS_PER_TURN,
    max_workers=int(os.environ.get("TW_MAX_WORKERS", "16")),
    sites_filter=SITES_FILTER,
    task_limit=TASK_LIMIT,
    rollout_mode=os.environ.get("TW_ROLLOUT_MODE", "accumulated"),
)

train_dataset = vf_env.get_dataset()
print(
    f">>> TimeWarp train tasks: {len(train_dataset)} "
    f"(IDs 104-231; sites={SITES_FILTER or 'all'}, limit={TASK_LIMIT})"
)

VLLM_SERVER_HOST = os.environ.get("TW_VLLM_SERVER_HOST", "0.0.0.0")
VLLM_SERVER_PORT = int(os.environ.get("TW_VLLM_SERVER_PORT", "8000"))
training_args = vf.get_default_grpo_config(
    run_name=RUN_NAME,
    num_gpus=NUM_GPUS,
    vllm_server_host=VLLM_SERVER_HOST,
    vllm_server_port=VLLM_SERVER_PORT,
    num_generations=(NUM_GENERATIONS if NUM_GENERATIONS > 0 else None),
)
training_args.output_dir = OUTPUT_DIR

training_args.max_prompt_length = int(os.environ.get("TW_MAX_PROMPT_LENGTH", "16384"))

_mcl = os.environ.get("TW_MAX_COMPLETION_LENGTH")
if _mcl:
    training_args.max_completion_length = int(_mcl)
training_args.max_completion_length = MAX_TOKENS_PER_TURN

_vllm_util = os.environ.get("TW_VLLM_GPU_MEM_UTIL")
if _vllm_util:
    training_args.vllm_gpu_memory_utilization = float(_vllm_util)

if NUM_GENERATIONS > 0:
    training_args.num_generations = NUM_GENERATIONS
    training_args.per_device_train_batch_size = (
        PER_DEVICE_BATCH if PER_DEVICE_BATCH > 0 else NUM_GENERATIONS
    )
_num_train_procs = max(1, NUM_GPUS - 1)

if GRAD_ACCUM > 0:
    training_args.gradient_accumulation_steps = GRAD_ACCUM

_gen_batch = (
    training_args.per_device_train_batch_size
    * _num_train_procs
    * max(1, training_args.gradient_accumulation_steps)
)
assert training_args.num_generations >= 2, (
    f"GRPO needs num_generations >= 2 (got {training_args.num_generations})."
)
assert _gen_batch % training_args.num_generations == 0, (
    f"TRL requires generation_batch_size = per_device({training_args.per_device_train_batch_size}) "
    f"* num_train_procs({_num_train_procs}) * grad_accum({training_args.gradient_accumulation_steps}) "
    f"= {_gen_batch} divisible by num_generations ({training_args.num_generations})."
)

if training_args.gradient_checkpointing:
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

if BETA is not None:
    training_args.beta = float(BETA)

_bt = os.environ.get("TW_BETA_THINK")
_ba = os.environ.get("TW_BETA_ACTION")
if _bt is not None or _ba is not None:
    _forced = training_args.beta if training_args.beta else 5e-4
    training_args.beta = _forced
    print(
        f">>> [TAG 1d] role-decoupled KL: args.beta={_forced} (scalar, preserved) "
        f"so the ref-model forward runs; per-role think={_bt}, action={_ba}, "
        f"unset-role default={_forced}"
    )

_lr = os.environ.get("TW_LEARNING_RATE")
if _lr:
    training_args.learning_rate = float(_lr)
_niter = int(os.environ.get("TW_NUM_ITERATIONS", "-1"))
if _niter > 0:
    training_args.num_iterations = _niter
_ltype = os.environ.get("TW_LOSS_TYPE")
if _ltype:
    if _ltype.strip().lower() in ("turn", "turn_const", "mgrpo"):
        training_args.loss_type = "dr_grpo"
    else:
        training_args.loss_type = _ltype
_srew = os.environ.get("TW_SCALE_REWARDS")
if _srew:
    _m = _srew.strip().lower()
    training_args.scale_rewards = {
        "false": "none",
        "0": "none",
        "none": "none",
        "off": "none",
        "true": "group",
        "1": "group",
        "group": "group",
        "batch": "batch",
    }.get(_m, _srew)
_wr = os.environ.get("TW_WARMUP_RATIO")
if _wr:
    training_args.warmup_ratio = float(_wr)

_eps = os.environ.get("TW_EPSILON")
if _eps:
    training_args.epsilon = float(_eps)
_epsh = os.environ.get("TW_EPSILON_HIGH")
if _epsh:
    training_args.epsilon_high = float(_epsh)
_mtc = os.environ.get("TW_MASK_TRUNCATED")
if _mtc:
    training_args.mask_truncated_completions = _mtc.strip().lower() in ("1", "true", "yes", "on")
_temp = os.environ.get("TW_TEMPERATURE")
if _temp:
    training_args.temperature = float(_temp)
_topp = os.environ.get("TW_TOP_P")
if _topp:
    training_args.top_p = float(_topp)
print(
    f">>> scale_rewards={training_args.scale_rewards} "
    f"epsilon={training_args.epsilon} epsilon_high={getattr(training_args, 'epsilon_high', None)} "
    f"mask_truncated={training_args.mask_truncated_completions} "
    f"temperature={training_args.temperature} top_p={training_args.top_p} "
    f"dynamic_sampling={os.environ.get('TW_DYNAMIC_SAMPLING', '0')} "
    f"shaped_reward={os.environ.get('TW_SHAPED_REWARD', '0')}"
)

if GRAD_ACCUM > 0:
    training_args.gradient_accumulation_steps = GRAD_ACCUM
if TRAIN_MAX_STEPS > 0:
    training_args.max_steps = TRAIN_MAX_STEPS
if REPORT_TO is not None:
    training_args.report_to = [] if REPORT_TO.lower() in ("none", "") else [REPORT_TO]
if SAVE_STRATEGY is not None:
    training_args.save_strategy = SAVE_STRATEGY
NUM_TRAIN_EPOCHS = float(os.environ.get("TW_NUM_TRAIN_EPOCHS", "1"))
if NUM_TRAIN_EPOCHS != 1:
    training_args.num_train_epochs = NUM_TRAIN_EPOCHS

if os.environ.get("TW_LOG_COMPLETIONS") in ("0", "false", "False"):
    training_args.log_completions = False
print(
    f">>> grad_accum={training_args.gradient_accumulation_steps} "
    f"max_steps={training_args.max_steps} num_generations={training_args.num_generations} "
    f"per_device_train_batch_size={training_args.per_device_train_batch_size} "
    f"num_train_procs={_num_train_procs} "
    f"report_to={training_args.report_to} save_strategy={training_args.save_strategy}"
)

print(
    f">>> Training: loss_type={os.environ.get('TW_LOSS_TYPE', '(dr_grpo/base)')} "
    f"is_level={os.environ.get('TW_IS_LEVEL', 'token')} "
    f"turn_decay={os.environ.get('TW_TURN_DECAY', '1.0')} "
    f"turn_tmax={os.environ.get('TW_TURN_TMAX', str(MAX_STEPS))} "
    f"turn_p={os.environ.get('TW_TURN_P', '1.0')} "
    f"turn_floor={os.environ.get('TW_TURN_NORM_FLOOR', '0')} "
    f"beta_think={os.environ.get('TW_BETA_THINK', '(unset)')} "
    f"beta_action={os.environ.get('TW_BETA_ACTION', '(unset)')} "
    f"scale_rewards_impl={os.environ.get('TW_SCALE_REWARDS_IMPL', 'group')} "
    f"beta={training_args.beta}"
)


class GradClipTelemetryCallback(TrainerCallback):
    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = float(max_grad_norm)
        self.n = 0
        self.n_binding = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        gn = logs.get("grad_norm")
        if gn is None:
            return
        try:
            gn = float(gn)
        except (TypeError, ValueError):
            return
        self.n += 1
        binding = gn > self.max_grad_norm * (1.0 + 1e-6)
        if binding:
            self.n_binding += 1
        frac = self.n_binding / max(1, self.n)
        print(
            f">>> [clip-telemetry] step={state.global_step} grad_norm={gn:.4g} "
            f"binding={binding} frac_binding={frac:.3f} (max_grad_norm={self.max_grad_norm})",
            flush=True,
        )


trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=vf_env.get_rubric(),
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.add_callback(GradClipTelemetryCallback(training_args.max_grad_norm))

if __name__ == "__main__":
    if os.environ.get("TW_DIAG") == "1":
        import torch
        import torch.distributed as dist

        acc = trainer.accelerator
        cur = torch.cuda.current_device() if torch.cuda.is_available() else -1
        rank = dist.get_rank() if dist.is_initialized() else -1
        ws = dist.get_world_size() if dist.is_initialized() else -1

        try:
            from torch.distributed.distributed_c10d import _world

            n_pg = len(getattr(_world, "pg_map", {}))
            gcount = getattr(_world, "group_count", "n/a")
        except Exception as e:
            n_pg, gcount = f"err:{e}", "n/a"
        print(
            f">>> [DIAG rank?] cuda.current_device={cur} acc.device={acc.device} "
            f"acc.process_index={acc.process_index} acc.local={acc.local_process_index} "
            f"dist.init={dist.is_initialized()} dist.rank={rank} ws={ws} "
            f"n_process_groups={n_pg} group_count={gcount}",
            flush=True,
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(acc.local_process_index)
            print(
                f">>> [DIAG r{rank}] set_device({acc.local_process_index}); "
                f"now current={torch.cuda.current_device()}",
                flush=True,
            )
        if dist.is_initialized():
            t = torch.ones(1, device=f"cuda:{acc.local_process_index}")
            dist.barrier()
            dist.all_reduce(t)
            print(f">>> [DIAG r{rank}] barrier+all_reduce OK -> {t.item()}", flush=True)

        n_total = sum(1 for _ in trainer.model.parameters())
        n_dsid = sum(1 for p in trainer.model.parameters() if hasattr(p, "ds_id"))
        print(f">>> [DIAG r{rank}] model.params total={n_total} with_ds_id={n_dsid}", flush=True)
    trainer.train()
