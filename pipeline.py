
from typing import Any, Optional, Tuple


def load_ppl(ppl_config: Any) -> Any:
    """Create and return a model based on the provided configuration.
    This helper builds model config objects and model instances for supported
    backbones. Return shapes are intentionally kept compatible with the
    existing codebase:
      - For 'ijepa': returns (model, processor)
      - For 'mae'  : returns model
    The function performs basic validation of expected attributes on
    `ppl_config` and raises informative errors when required fields are
    missing. It intentionally does not download or modify pretrained weights
    beyond what the underlying transformers classes may do.
    Args:
        ppl_config: object/namespace containing model configuration fields
            (e.g. name, img_size, patch_size, chans, D, layers, heads, mlp_ratio, id,
            enc_layers, enc_heads, enc_D, dec_layers, dec_heads, dec_D).
    Returns:
        Model instance, or (model, processor) for IJepa.
    Raises:
        ValueError: when required attributes are missing or model name is unknown.
        NotImplementedError: for backbones that are intentionally unimplemented.
    """

    # Ensure we have a name to switch on
    name = getattr(ppl_config, "name", None).lower()
    if not name:
        raise ValueError("ppl_config must have a 'name' attribute (e.g. 'ijepa' or 'mae')")

    # IJepa: requires a processor + model config
    if name == "ijepa":
        try:
            import importlib
            transformers = importlib.import_module("transformers")
            AutoProcessor = getattr(transformers, "AutoProcessor")
            IJepaConfig = getattr(transformers, "IJepaConfig")
            IJepaModel = getattr(transformers, "IJepaModel")
        except Exception as e:
            raise ImportError(
                "The 'transformers' package is required for the 'ijepa' backend. "
                "Install it with: pip install transformers"
            ) from e
        # Validate required fields and provide clearer error messages
        required = ["img_size", "patch_size", "chans", "D", "layers", "heads", "mlp_ratio", "id"]
        missing = [f for f in required if not hasattr(ppl_config, f)]
        if missing:
            raise ValueError(f"Missing required ijepa config fields: {missing}")
        cfg = IJepaConfig(
            image_size=ppl_config.img_size,
            patch_size=ppl_config.patch_size,
            num_channels=ppl_config.chans,
            hidden_size=ppl_config.D,
            num_hidden_layers=ppl_config.layers,
            num_attention_heads=ppl_config.heads,
            mlp_ratio=ppl_config.mlp_ratio,
        )
        # Instantiate model from config (no pretrained weights by default)
        ppl = IJepaModel(cfg)
        # Load processor from a pretrained identifier. Let underlying
        # transformers raise an error if the id is invalid / missing.
        processor = AutoProcessor.from_pretrained(ppl_config.id)
        return ppl, processor

    # MAE: encoder/decoder config
    elif name == "mae":
        try:
            transformers = importlib.import_module("transformers")
            MaeConfig = getattr(transformers, "MaeConfig")
            MaeModel = getattr(transformers, "MaeModel")
        except Exception as e:
            raise ImportError(
                "The 'transformers' package is required for the 'mae' backend. "
                "Install it with: pip install transformers"
            ) from e
        required = [
            "img_size",
            "patch_size",
            "chans",
            "enc_layers",
            "enc_heads",
            "enc_D",
            "dec_layers",
            "dec_heads",
            "dec_D",
            "mlp_ratio",
        ]
        missing = [f for f in required if not hasattr(ppl_config, f)]
        if missing:
            raise ValueError(f"Missing required mae config fields: {missing}")
        cfg = MaeConfig(
            image_size=ppl_config.img_size,
            patch_size=ppl_config.patch_size,
            num_channels=ppl_config.chans,
            encoder_layers=ppl_config.enc_layers,
            encoder_attention_heads=ppl_config.enc_heads,
            encoder_hidden_size=ppl_config.enc_D,
            decoder_layers=ppl_config.dec_layers,
            decoder_attention_heads=ppl_config.dec_heads,
            decoder_hidden_size=ppl_config.dec_D,
            mask_ratio=ppl_config.mlp_ratio,
        )
        ppl = MaeModel(cfg)
        return ppl
    elif name == "simgroupattn":
        raise NotImplementedError(
            "'simgroupattn' backend is not implemented in load_ppl(). "
            "Please implement model construction here or call the project's "
            "factory function for simgroupattn instead."
        )

    # Unknown model name
    else:
        raise ValueError(f"Unknown model name: {name}")