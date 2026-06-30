"""CAFREC — Context-Adaptive Fusion recommender (Figures 3 & 4 of the plan).

Architecture
------------
  z_short = SASRec-style self-attentive encoder over the current session
  z_long  = projected, precomputed LLM temporal-profile embedding (per user)
  g       = sigmoid MLP over EXOGENOUS session-context features  (the novelty)
  z       = g * z_long + (1 - g) * z_short        (element-wise fusion)

The short-term encoder is identical to RecBole's SASRec on purpose: it makes
the gating the only thing that differs from the SASRec baseline, so the
comparison attributes any gain to context-adaptive fusion rather than to a
different encoder.

Runs-before-its-data design
---------------------------
  * LLM profile: loaded & frozen from `llm_profile_path` if given, else a
    learnable embedding stand-in (so it runs before T2.2 profiles exist).
  * Context features: pulled from `context_fields` in the interaction if those
    columns exist (T2.1), else a fallback that uses session length only.
This lets the model train on ml-100k today and become fully CAFREC once the
feature columns and profiles are in place.

Ablations (config `ablation`, for T3.1)
  none         full CAFREC
  no_profiler  z_long := 0   (isolates the LLM profiler)
  static_gate  g := sigmoid(scalar), context-independent  (isolates conditioning)
  concat       z := W[z_long ; z_short]   (isolates gated vs concat fusion)
"""
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class CAFREC(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.n_users = dataset.user_num
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.ablation = config["ablation"]

        # --- shared item embedding (also the scoring head) ------------------
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)

        # --- short-term path: SASRec self-attentive encoder ----------------
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            hidden_size=self.hidden_size,
            inner_size=config["inner_size"],
            hidden_dropout_prob=config["hidden_dropout_prob"],
            attn_dropout_prob=config["attn_dropout_prob"],
            hidden_act=config["hidden_act"],
            layer_norm_eps=config["layer_norm_eps"],
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

        # --- long-term path: LLM temporal profile --------------------------
        self.profile_dim = config["profile_dim"]
        self.user_profile = nn.Embedding(self.n_users, self.profile_dim)
        self.profile_proj = nn.Linear(self.profile_dim, self.hidden_size)

        # --- context-adaptive gating (Figure 4) ----------------------------
        self.context_fields = config["context_fields"]      # [] -> fallback path
        self.n_context_features = config["n_context_features"]
        gh = config["gating_hidden"]
        self.gating_mlp = nn.Sequential(
            nn.Linear(self.n_context_features, gh), nn.ReLU(),
            nn.Linear(gh, gh), nn.ReLU(),
            nn.Linear(gh, self.hidden_size), nn.Sigmoid(),   # g in [0,1]^H
        )
        # ablation helpers (cheap; created always, used only when selected)
        self.static_gate = nn.Parameter(torch.zeros(1))      # sigmoid(0)=0.5
        self.concat_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.initializer_range = config["initializer_range"]
        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            raise NotImplementedError("loss_type must be 'CE' or 'BPR'")

        self.apply(self._init_weights)
        self._maybe_load_profiles(config.get("llm_profile_path"))

    # ----------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _maybe_load_profiles(self, path):
        """Load frozen LLM profile vectors [n_users, profile_dim] if provided."""
        if not path:
            return  # learnable stand-in until T2.2 profiles exist
        profiles = torch.load(path)
        if profiles.shape != (self.n_users, self.profile_dim):
            raise ValueError(
                f"profile tensor {tuple(profiles.shape)} != "
                f"({self.n_users}, {self.profile_dim})"
            )
        self.user_profile.weight.data.copy_(profiles)
        self.user_profile.weight.requires_grad_(False)

    # ----- the three paths -------------------------------------------------
    def _short_term(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long,
                                    device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        input_emb = self.dropout(self.LayerNorm(input_emb))
        mask = self.get_attention_mask(item_seq)             # causal mask
        trm_output = self.trm_encoder(input_emb, mask, output_all_encoded_layers=True)
        output = self.gather_indexes(trm_output[-1], item_seq_len - 1)
        return output                                        # z_short [B, H]

    def _long_term(self, user):
        return self.profile_proj(self.user_profile(user))    # z_long  [B, H]

    def _build_context(self, interaction, item_seq_len):
        """Per-request context vector x_ctx [B, n_context_features]."""
        if self.context_fields and all(f in interaction for f in self.context_fields):
            cols = [interaction[f].float().unsqueeze(-1) for f in self.context_fields]
            return torch.cat(cols, dim=-1)
        # fallback: only session length is always available
        b = item_seq_len.size(0)
        ctx = torch.zeros(b, self.n_context_features, device=item_seq_len.device)
        ctx[:, 0] = item_seq_len.float() / self.max_seq_length
        return ctx

    def _gate(self, context, ref):
        if self.ablation == "static_gate":
            return torch.sigmoid(self.static_gate).expand(ref.size(0), self.hidden_size)
        return self.gating_mlp(context)                      # g [B, H]

    # ----- fused representation -------------------------------------------
    def _fuse(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        z_short = self._short_term(item_seq, item_seq_len)
        z_long = self._long_term(interaction[self.USER_ID])
        if self.ablation == "no_profiler":
            z_long = torch.zeros_like(z_long)
        if self.ablation == "concat":
            return self.concat_proj(torch.cat([z_long, z_short], dim=-1))
        g = self._gate(self._build_context(interaction, item_seq_len), z_short)
        return g * z_long + (1.0 - g) * z_short              # z [B, H]

    # ----- RecBole interface ----------------------------------------------
    def calculate_loss(self, interaction):
        seq_output = self._fuse(interaction)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "CE":
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            return self.loss_fct(logits, pos_items)
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_score = (seq_output * self.item_embedding(pos_items)).sum(dim=-1)
        neg_score = (seq_output * self.item_embedding(neg_items)).sum(dim=-1)
        return self.loss_fct(pos_score, neg_score)

    def predict(self, interaction):
        seq_output = self._fuse(interaction)
        test_item_emb = self.item_embedding(interaction[self.ITEM_ID])
        return (seq_output * test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        seq_output = self._fuse(interaction)
        return torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
