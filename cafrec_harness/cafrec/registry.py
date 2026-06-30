"""Model registry — the single place you add a model to the comparison.

A model is registered under a string KEY. The KEY is the only thing the
runner needs: built-in RecBole models point at a *name* string, your own
models point at a *class*. Everything downstream is identical, so a custom
model (CAFREC) slots in exactly like a baseline.

Each spec also carries its training "contract": the loss type and whether
negative sampling is on. This is where the SASRec(CE)/HGN(BPR) asymmetry
lives once, so callers never set it by hand.
"""
from dataclasses import dataclass, field
from typing import Optional, Type, Union


# --- Training contracts -------------------------------------------------------
# Cross-entropy: full-softmax over the catalogue, NO negative sampling.
CE = {"loss_type": "CE", "train_neg_sample_args": None}
# Bayesian Personalised Ranking: pairwise, needs one uniform negative.
BPR = {"loss_type": "BPR",
       "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1}}


@dataclass
class ModelSpec:
    """How to build and train one model.

    model:    a RecBole model NAME (str) for built-ins, or a model CLASS
              for your own. The runner branches on this type.
    contract: loss_type / train_neg_sample_args (use CE or BPR above).
    config:   model-specific hyperparameters merged into the RecBole config.
              Required for custom classes (no auto-loaded MODEL.yaml exists).
    trainer:  optional custom Trainer subclass; default trainer is used if None.
    """
    model: Union[str, Type]
    contract: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    trainer: Optional[Type] = None


def _build_registry():
    # Imported lazily so the registry module stays importable even if a model
    # file has a heavy/optional dependency.
    from cafrec.models.cafrec import CAFREC

    return {
        # --- Baselines (resolved by RecBole name) ---------------------------
        "SASRec": ModelSpec(model="SASRec", contract=CE),   # strong sequential anchor
        "HGN":    ModelSpec(model="HGN",    contract=BPR),  # long+short via gating
        # "SHAN":  ModelSpec(model="SHAN",  contract=BPR),  # uncomment to add a 3rd

        # --- Your model (resolved by CLASS) ---------------------------------
        # Note it registers identically to the baselines: same key->spec shape.
        # A custom class has no auto-loaded model yaml, so all hyperparameters
        # live here.
        "CAFREC": ModelSpec(
            model=CAFREC,
            contract=CE,
            config={
                # short-term encoder (SASRec defaults -> matches the baseline)
                "hidden_size": 64, "n_layers": 2, "n_heads": 2, "inner_size": 256,
                "hidden_dropout_prob": 0.5, "attn_dropout_prob": 0.5,
                "hidden_act": "gelu", "layer_norm_eps": 1e-12,
                "initializer_range": 0.02,
                # long-term LLM profile (path=None -> learnable stand-in for now)
                "profile_dim": 64, "llm_profile_path": None,
                # context-adaptive gating.
                # context_fields=[] -> fallback (session length only) so it runs
                # on ml-100k. Once T2.1 writes the columns, set e.g.
                #   context_fields: [session_len, dwell_entropy, cat_drift,
                #                    request_source, intent_label]
                # and n_context_features to its length (4 in the submitted plan,
                # 5 with the reasoning-intent label -- reconcile this with the doc).
                "n_context_features": 5, "context_fields": [], "gating_hidden": 64,
                # ablation: none | no_profiler | static_gate | concat   (T3.1)
                "ablation": "none",
            },
        ),
    }


MODEL_REGISTRY = _build_registry()


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown model key '{key}'. "
            f"Registered: {sorted(MODEL_REGISTRY)}"
        )


def list_models():
    return sorted(MODEL_REGISTRY)
