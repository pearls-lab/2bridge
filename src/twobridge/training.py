"""Single-device and torchrun/DDP training with resumable, self-contained checkpoints."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .backend import build_model
from .dataset import LeRobotData
from .preprocessing import Statistics, PromptTokenizer, compute_statistics, prepare_batch, file_sha256


def learning_rate(step, cfg):
    if cfg.warmup_steps and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    fraction = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps - 1))
    return cfg.end_lr + (cfg.lr - cfg.end_lr) * (1 + math.cos(math.pi * fraction)) / 2


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def rng_state():
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(
    output, model, optimizer, config, stats, step, epoch, batch_offset, dataset_description, rank, world_size
):
    states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()
    if rank == 0:
        path = output / "checkpoints" / f"step_{step:08d}"
        temporary = path.with_name(path.name + ".tmp")
        temporary.mkdir(parents=True, exist_ok=False)
        safetensors.torch.save_model(unwrap(model), str(temporary / "model.safetensors"))
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "epoch": epoch,
                "batch_offset": batch_offset,
                "rng": states,
                "world_size": world_size,
            },
            temporary / "training_state.pt",
        )
        stats.save(temporary / "stats.json")
        shutil.copyfile(config.training.tokenizer, temporary / "tokenizer.model")
        (temporary / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (temporary / "dataset.json").write_text(json.dumps(dataset_description, indent=2) + "\n")
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "step": step,
                    "torch": str(torch.__version__),
                    "tokenizer_sha256": file_sha256(temporary / "tokenizer.model"),
                },
                indent=2,
            )
            + "\n"
        )
        temporary.rename(path)
        print(json.dumps({"checkpoint": str(path)}), flush=True)
    if world_size > 1:
        dist.barrier()


@torch.no_grad()
def evaluate(model, loader, stats, tokenizer, device, max_batches):
    state = rng_state()
    model.eval()
    total, count = 0.0, 0
    try:
        # Fixed validation noise, independent of how frequently validation runs.
        torch.manual_seed(0)
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            obs, actions = prepare_batch(batch, stats, tokenizer, device)
            loss = unwrap(model)(obs, actions)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite validation loss")
            size = actions.shape[0]
            total += float(loss) * size
            count += size
    finally:
        model.train()
        restore_rng(state)
    return total / max(1, count)


def run(config, resume=None):
    config.validate()
    t = config.training
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and t.precision != "float32":
        raise ValueError(
            "CPU execution requires training.precision='float32'; full Pi0.5 training needs a GPU"
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if t.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("Selected GPU does not support bfloat16")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        _run(config, resume, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(config, resume, device, rank, world):
    t = config.training
    train = LeRobotData(config)
    val = LeRobotData(config, "val") if config.dataset.val_episodes else None
    description = train.describe()
    description["val_episodes"] = config.dataset.val_episodes
    # Fingerprint numeric data, instructions, metadata and file identities for resume.
    import hashlib

    h = hashlib.sha256()
    h.update((train.root / "meta/info.json").read_bytes())
    for array in (train.states, train.actions):
        h.update(array.tobytes())
    for ep, frame in train.samples:
        h.update(json.dumps([ep, frame, train.prompt(train.episodes[ep][frame])]).encode())
    h.update(json.dumps(train.episode_metadata, sort_keys=True).encode())
    for path in train.paths:
        st = path.stat()
        h.update(f"{path.relative_to(train.root)}:{st.st_size}:{st.st_mtime_ns}".encode())
    description["data_fingerprint"] = h.hexdigest()
    if len(train) < t.batch_size * world:
        raise ValueError("Training split is smaller than one global batch; reduce batch_size")
    # Decode and tokenize before model allocation, including all configured cameras.
    train[0]
    output = Path(t.output)
    resume = Path(resume).resolve() if resume else None
    weights = Path(t.weights)
    if weights.is_dir():
        weights = weights / "model.safetensors"
    if not resume and not weights.is_file():
        raise FileNotFoundError(
            "Set training.weights to pretrained OpenPI PyTorch model.safetensors (or its directory)"
        )
    if not Path(t.tokenizer).is_file():
        raise FileNotFoundError("Set training.tokenizer to the PaliGemma sentencepiece tokenizer.model")
    saved = None
    if resume:
        if resume.parent.parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to training.output/checkpoints")
        if json.loads((resume / "config.json").read_text()) != config.to_dict():
            raise ValueError(
                "Resume configuration differs; resume requires the original training configuration"
            )
        if json.loads((resume / "dataset.json").read_text()) != description:
            raise ValueError("Dataset changed since checkpoint; refusing inconsistent resume")
        if file_sha256(t.tokenizer) != file_sha256(resume / "tokenizer.model"):
            raise ValueError("Tokenizer changed since checkpoint")
        # Only our local, explicitly selected training checkpoint is deserialized.
        saved = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        if saved["world_size"] != world:
            raise ValueError("Resume requires the same world size")
        if saved["step"] >= t.steps:
            raise ValueError("This checkpoint has already completed the configured training steps")
        later = [
            p
            for p in (output / "checkpoints").glob("step_*")
            if p.name > resume.name and not p.name.endswith(".tmp")
        ]
        if later:
            raise ValueError("Resume from the newest checkpoint; later checkpoints already exist")
        stats = Statistics.load(resume / "stats.json")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output is not empty: {output}; use --resume or a new output")
        stats = compute_statistics(train)
    tokenizer = PromptTokenizer(t.tokenizer, t.max_token_len)
    for index in np.linspace(0, len(train) - 1, min(len(train), 32), dtype=int):
        ep, frame = train.samples[index]
        tokenizer.encode(
            train.prompt(train.episodes[ep][frame]), stats.normalize(train.numeric(index)[0], "state")
        )
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {"dataset": description, "global_batch_size": t.batch_size * world * t.accumulation_steps}
            ),
            flush=True,
        )
    if world > 1:
        dist.barrier()
    torch.manual_seed(t.seed)
    np.random.seed(t.seed)
    model = build_model(t, train.action_dim)
    load_path = resume / "model.safetensors" if resume else weights
    safetensors.torch.load_model(model, str(load_path), strict=True)
    model.to(device)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=t.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=t.weight_decay
    )
    step, epoch, batch_offset = 0, 0, 0
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, batch_offset = saved["step"], saved["epoch"], saved["batch_offset"]
        restore_rng(saved["rng"][rank])
    else:
        torch.manual_seed(t.seed + rank)
    sampler = DistributedSampler(
        train, num_replicas=world, rank=rank, shuffle=True, seed=t.seed, drop_last=True
    )
    loader = DataLoader(
        train,
        batch_size=t.batch_size,
        sampler=sampler,
        num_workers=t.workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(t.seed),
        multiprocessing_context="spawn" if t.workers else None,
    )
    val_loader = (
        DataLoader(
            val,
            batch_size=t.batch_size,
            shuffle=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(t.seed),
        )
        if val
        else None
    )
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(batch_offset):
        next(iterator)
    model.train()
    started = time.monotonic()
    while step < t.steps:
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for micro in range(t.accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch, batch_offset = epoch + 1, 0
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch_offset += 1
            obs, actions = prepare_batch(batch, stats, tokenizer, device)
            context = model.no_sync() if world > 1 and micro < t.accumulation_steps - 1 else nullcontext()
            with context:
                losses = model(obs, actions)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"Non-finite training loss at step {step}")
                (losses["loss"] / t.accumulation_steps).backward()
            for key, value in losses.items():
                metrics[key] = metrics.get(key, 0.0) + float(value.detach()) / t.accumulation_steps
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip, error_if_nonfinite=True)
        lr = learning_rate(step, t)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1
        metrics.update(step=step, lr=lr, grad_norm=float(norm), elapsed_seconds=time.monotonic() - started)
        if rank == 0 and (step % t.log_every == 0 or step == 1):
            print(json.dumps(metrics), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics) + "\n")
        if val_loader and step % t.eval_every == 0:
            # Every rank evaluates identical held-out frames; no DDP forward collectives.
            value = evaluate(model, val_loader, stats, tokenizer, device, t.eval_batches)
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"step": step, "val_loss": value}) + "\n")
                print(json.dumps({"step": step, "val_loss": value}), flush=True)
        if step % t.save_every == 0 or step == t.steps:
            save_checkpoint(
                output, model, optimizer, config, stats, step, epoch, batch_offset, description, rank, world
            )
