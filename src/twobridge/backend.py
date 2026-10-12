"""Load the pinned OpenPI backend without modifying installed package files."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_PATCHED = False


def prepare_transformers():
    global _PATCHED
    if _PATCHED:
        return
    import transformers

    if transformers.__version__ != "4.53.2":
        raise RuntimeError(f"This backend requires transformers==4.53.2, found {transformers.__version__}")
    root = Path(__file__).parent / "_vendor/transformers_replace/models"
    # Load into the expected module names so upstream relative imports resolve.
    # These are process-local replacements, never writes into site-packages.
    for relative in (
        "gemma/configuration_gemma.py",
        "gemma/modeling_gemma.py",
        "siglip/modeling_siglip.py",
        "paligemma/modeling_paligemma.py",
    ):
        model, filename = relative.split("/")
        name = f"transformers.models.{model}.{filename[:-3]}"
        parent = importlib.import_module(f"transformers.models.{model}")
        spec = importlib.util.spec_from_file_location(name, root / relative)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(parent, filename[:-3], module)
        for key, value in vars(module).items():
            if isinstance(value, type) and value.__module__ == name:
                setattr(parent, key, value)
                setattr(transformers, key, value)
    # AutoConfig/AutoModel must resolve the same configuration class identities.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.gemma.configuration_gemma import GemmaConfig

    CONFIG_MAPPING.register("gemma", GemmaConfig, exist_ok=True)
    _PATCHED = True


def build_model(training, action_dim):
    prepare_transformers()
    from ._vendor.pi0_pytorch import PI0Pytorch

    config = SimpleNamespace(
        pi05=True,
        action_dim=32,
        action_horizon=training.horizon,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype=training.precision,
        pytorch_compile_mode=None,
        real_action_dim=action_dim,
        deriv_dims=training.deriv_dims,
        deriv_weight=training.deriv_weight,
    )
    model = PI0Pytorch(config)
    if training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model
