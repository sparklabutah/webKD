from trl import GRPOConfig
from typing import List, Optional


def get_default_grpo_config(
    run_name: str,
    num_gpus: int = 1,
    reward_weights: Optional[List[float]] = None,
    vllm_server_host: str = "0.0.0.0",
    vllm_server_port: int = 8000,
    num_generations: Optional[int] = None,
) -> GRPOConfig:

    if num_generations is None:
        num_generations = max(2, num_gpus - 1)
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant",
        warmup_steps=0,
        num_train_epochs=1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.01,
        num_iterations=1,
        beta=0.001,
        max_prompt_length=16382,
        max_completion_length=1024,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_generations=num_generations,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=0.125,
        save_only_model=True,
        eval_strategy="no",
        eval_steps=0.125,
        eval_on_start=False,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        loss_type="grpo",
        scale_rewards="group",
        vllm_importance_sampling_correction=False,
        importance_sampling_level="token",
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="none",
        reward_weights=reward_weights,
        temperature=1.0,
    )
