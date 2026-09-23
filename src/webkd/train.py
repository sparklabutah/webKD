"""Prepare a portable LLaMA-Factory training configuration and launch it."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import yaml

from webkd.weights import load_conversations


def validate_sidecar(path, conversations, role=None):
    """Reject a stale or incomplete weight file before starting training."""
    with open(path) as handle:
        blob = json.load(handle)
    if role and blob.get("kl_role_mask") != role:
        raise ValueError(f"WebKD requires a {role} anchor mask built with build-anchor")
    entries = blob.get("entries", {})
    for row in conversations:
        entry = entries.get(row["sha1"])
        if entry is None or entry.get("len") != len(row["content"]):
            raise ValueError(f"Missing or stale sidecar entry for example {row['index']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("webkd", "sft"), default="webkd")
    parser.add_argument("--model", required=True, help="Student checkpoint or model identifier")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", required=True, help="LLaMA-Factory chat template")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--reference", help="Frozen student checkpoint; defaults to --model")
    parser.add_argument("--kl-beta", type=float, default=0.03)
    parser.add_argument(
        "--dry-run", action="store_true", help="Write config without loading models"
    )
    args = parser.parse_args()
    conversations = load_conversations(args.dataset)
    if not conversations:
        parser.error("The dataset contains no assistant targets")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("WEBKD_TOKEN", "WEBKD_KL", "WEBKD_SURGICAL"))
    }
    if args.method == "webkd":
        if not args.weights or not args.anchor:
            parser.error("WebKD requires both --weights and --anchor")
        if not math.isfinite(args.kl_beta) or args.kl_beta <= 0:
            parser.error("--kl-beta must be finite and positive")
        validate_sidecar(args.weights, conversations)
        validate_sidecar(args.anchor, conversations, role="rationale")
        env.update(
            WEBKD_TOKEN_WEIGHTS=str(args.weights.resolve()),
            WEBKD_KL_GAMMA_WEIGHTS=str(args.anchor.resolve()),
            WEBKD_KL_REF=args.reference or args.model,
            WEBKD_KL_BETA=str(args.kl_beta),
        )
    elif args.weights or args.anchor:
        parser.error("Use the WebKD method when supplying weights and an anchor")
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        parser.error("--config must contain a YAML mapping")
    if any(config.get(key) for key in ("packing", "mask_history", "train_on_prompt")):
        parser.error("Disable packing, mask_history, and train_on_prompt")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data_dir = destination / "training_data"
    data_dir.mkdir(exist_ok=True)
    dataset_link = data_dir / "corpus.json"
    if dataset_link.is_symlink():
        dataset_link.unlink()
    elif dataset_link.exists():
        parser.error(f"Refusing to replace existing file: {dataset_link}")
    dataset_link.symlink_to(args.dataset.resolve())
    info = {
        "corpus": {
            "file_name": "corpus.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
    }
    (data_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    config.update(
        model_name_or_path=args.model,
        dataset="corpus",
        dataset_dir=str(data_dir),
        output_dir=str(destination),
        template=args.template,
    )
    config_path = destination / "training.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    # Save only method options; credentials and other environment variables stay private.
    options = {
        key: value for key, value in env.items() if key.startswith(("WEBKD_TOKEN", "WEBKD_KL"))
    }
    (destination / "method.json").write_text(json.dumps(options, indent=2) + "\n")
    if args.dry_run:
        print(f"Prepared {args.method} configuration: {config_path}")
        return
    launcher = shutil.which("llamafactory-cli") or shutil.which("lmf")
    if not launcher:
        parser.error("Install the LLaMA-Factory integration first; see docs/training.md")
    if args.method == "webkd":
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

        if not hasattr(CustomSeq2SeqTrainer, "_webkd_kl_compute_loss"):
            parser.error("The installed LLaMA-Factory is missing the WebKD integration")
    subprocess.run([launcher, "train", str(config_path)], env=env, check=True)


if __name__ == "__main__":
    main()
