"""Public command-line entry points."""

import argparse
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="2bridge: Pi0.5 fine-tuning on LeRobot datasets")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "train"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "inspect":
            p.add_argument("--samples", type=int, default=3, help="Number of evenly spaced frames to decode")
        else:
            p.add_argument("--resume", help="Path to a complete step checkpoint")
    p = commands.add_parser("download", help="Explicitly download a dataset from the Hugging Face Hub")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--revision", required=True, help="Pinned dataset revision (prefer a commit SHA)")
    args = parser.parse_args()
    if args.command == "download":
        from huggingface_hub import snapshot_download

        snapshot_download(args.repo_id, repo_type="dataset", revision=args.revision, local_dir=args.root)
        return
    config = load_config(args.config)
    if args.command == "train":
        from .training import run

        run(config, args.resume)
        return
    import numpy as np
    from .dataset import LeRobotData
    from .preprocessing import compute_statistics, prepare_batch, PromptTokenizer
    from torch.utils.data import default_collate
    import torch

    if args.samples < 1:
        parser.error("--samples must be positive")
    dataset = LeRobotData(config)
    statistics = compute_statistics(dataset)
    summary = dataset.describe()
    if config.dataset.val_episodes:
        summary["validation"] = LeRobotData(config, "val").describe()
    summary["decoded_samples"] = []
    tokenizer = (
        PromptTokenizer(config.training.tokenizer, config.training.max_token_len)
        if Path(config.training.tokenizer).is_file()
        else None
    )
    for i in np.linspace(0, len(dataset) - 1, min(args.samples, len(dataset)), dtype=int):
        item = dataset[int(i)]
        if tokenizer:
            prepare_batch(default_collate([item]), statistics, tokenizer, torch.device("cpu"))
        summary["decoded_samples"].append(
            {
                "index": int(i),
                "prompt": item["prompt"],
                "images": list(item["images"].shape),
                "valid_actions": int(item["action_mask"].sum()),
            }
        )
    summary["tokenizer_checked"] = tokenizer is not None
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
