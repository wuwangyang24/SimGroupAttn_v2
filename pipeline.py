from typing import Any
import importlib

# Mapping: model name → (ConfigClassName, ModelClassName)
_transformer_classes = {
    "ijepa": ("IJepaConfig", "IJepaModel"),
    "mae": ("MaeConfig", "MAEForPreTraining"),
    "simmim": ("SimMIMConfig", "SimMIMForPreTraining"),
    "data2vec": ("Data2VecVisionConfig", "Data2VecVisionModel"),
}

# Required fields per model
_required_fields = {
    "ijepa": ["img_size","patch_size","in_chans","embed_dim","layers","heads","mlp_ratio"],
    "mae": ["img_size","patch_size","chans","enc_layers","enc_heads","enc_D",
            "dec_layers","dec_heads","dec_D","mlp_ratio"],
    "simmim": ["img_size","patch_size","chans","D","layers","heads","inter_D","mlp_ratio"],
    "data2vec": ["img_size","patch_size","D","layers","heads"]
}

# Mapping from config kwargs → ppl_config attributes
_param_map = {
    "ijepa": {
        "image_size": "img_size",
        "patch_size": "patch_size",
        "num_channels": "in_chans",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
        "num_attention_heads": "heads",
        "mlp_ratio": "mlp_ratio"
    },
    "mae": {
        "image_size": "img_size",
        "patch_size": "patch_size",
        "num_channels": "chans",
        "encoder_layers": "enc_layers",
        "encoder_attention_heads": "enc_heads",
        "encoder_hidden_size": "enc_D",
        "decoder_layers": "dec_layers",
        "decoder_attention_heads": "dec_heads",
        "decoder_hidden_size": "dec_D",
        "mask_ratio": "mlp_ratio"
    },
    "simmim": {
        "image_size": "img_size",
        "patch_size": "patch_size",
        "num_channels": "chans",
        "hidden_size": "D",
        "num_hidden_layers": "layers",
        "num_attention_heads": "heads",
        "intermediate_size": "inter_D",
        "mask_ratio": "mlp_ratio"
    },
    "data2vec": {
        "image_size": "img_size",
        "patch_size": "patch_size",
        "hidden_size": "D",
        "num_hidden_layers": "layers",
        "num_attention_heads": "heads"
    }
}

def load_ppl(ppl_config: Any) -> Any:
    """
    Generic factory for vision models: IJepa, MAE, SimMIM, Data2Vec.
    """
    name = getattr(ppl_config, "name", None)
    if not name:
        raise ValueError("ppl_config must have a 'name' attribute (e.g. 'ijepa', 'mae', 'simmim', 'data2vec')")
    
    lname = name.lower()
    
    if lname not in _transformer_classes:
        raise ValueError(f"Unknown model name: {name}")
    
    # Validate required fields
    missing = [f for f in _required_fields[lname] if not hasattr(ppl_config, f)]
    if missing:
        raise ValueError(f"Missing required {lname} config fields: {missing}")
    
    # Import transformers and fetch classes
    try:
        transformers = importlib.import_module("transformers")
        cfg_class_name, model_class_name = _transformer_classes[lname]
        ConfigClass = getattr(transformers, cfg_class_name)
        ModelClass = getattr(transformers, model_class_name)
    except Exception as e:
        raise ImportError(
            f"The 'transformers' package is required for the '{name}' backend."
        ) from e
    
    # Map ppl_config -> config kwargs
    params = {cfg_arg: getattr(ppl_config, ppl_attr)
              for cfg_arg, ppl_attr in _param_map[lname].items()}
    
    cfg = ConfigClass(**params)
    model = ModelClass(cfg)
    return model
