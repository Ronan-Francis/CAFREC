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

Controls for the context-gated SASRec claim (both force z_long := 0, so each is
paired against no_profiler, i.e. CAFREC-NP):
  np_vector_gate   g := sigmoid(theta), theta in R^H, no context input
                   (is the gain just a learned per-dimension rescaling of z_short?)
  np_shuffled_ctx  x_ctx rows permuted within each batch, train AND eval
                   (does pairing the context with its own session matter?)
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
        # history-gated profiler (RON-45): give the gate a HISTORY-SUFFICIENCY
        # signal so it can suppress z_long when the history is too thin to profile
        # (H2 fix — the session-intent gate never saw history length). Modes:
        #   seqlen   append item_seq_len/max_seq_length  (saturates at the cap)
        #   logfull  append log1p(user total interactions)/norm  (unsaturated)
        #   suppress explicit z_long *= sigmoid((h_log - tau)/beta), tau/beta learnt
        # `history_gate=True` == seqlen (back-compat). Off by default -> unchanged.
        self.history_gate = bool(config["history_gate"])
        self.hg_mode = config["history_gate_mode"] or (
            "seqlen" if self.history_gate else None)
        self._append_hist = self.hg_mode in ("seqlen", "logfull")
        self._suppress = self.hg_mode == "suppress"
        gate_in = self.n_context_features + (1 if self._append_hist else 0)
        gh = config["gating_hidden"]
        self.gating_mlp = nn.Sequential(
            nn.Linear(gate_in, gh), nn.ReLU(),
            nn.Linear(gh, gh), nn.ReLU(),
            nn.Linear(gh, self.hidden_size), nn.Sigmoid(),   # g in [0,1]^H
        )
        # per-user activity level (total interactions) for logfull/suppress. This
        # is a user-level feature (history length), not target-dependent, so it
        # does not leak the held-out item. Built once from the dataset.
        if self.hg_mode in ("logfull", "suppress"):
            counts = torch.bincount(dataset.inter_feat[dataset.uid_field],
                                    minlength=self.n_users).float()
            self.register_buffer("user_hist_len", counts)
            self.register_buffer("log_hist_norm",
                                 torch.log1p(counts.max()).clamp(min=1.0))
        if self._suppress:                                   # thin-history damper
            self.hg_tau = nn.Parameter(torch.tensor(0.3))    # ~ threshold on h_log
            self.hg_beta = nn.Parameter(torch.tensor(0.15))  # softness
        # --- fusion FORM (RON-61) -----------------------------------------
        # z = z_short + g * (z_long - (1 - lambda) * z_short)
        #   lambda 0 -> convex   (incumbent: g*z_long + (1-g)*z_short)
        #   lambda 1 -> additive (profile enters as a pure correction, so g=0
        #                         recovers z_short exactly and the model can
        #                         keep the FULL session while adding profile —
        #                         a region the convex form cannot reach)
        self.fusion_mode = config["fusion_mode"] or "convex"
        if self.fusion_mode == "per_user":
            # free scalar per user; NOT context-conditioned, so a win here is
            # per-user personalisation of the fusion FORM, not context
            # adaptivity. No cold-start path for unseen users (fine under
            # leave-one-out, where every test user has train rows).
            self.fusion_lambda = nn.Embedding(self.n_users, 1)
        elif self.fusion_mode == "hybrid":
            # lambda_u = sigmoid( f(x_ctx) + b_u ): a context-conditioned
            # prediction PLUS a free per-user offset (a random-effects model).
            # Unlike per_user this says WHY a user got its value — the feature
            # head generalises, the bias absorbs what the features miss — which
            # is what makes the cohort analysis interpretable rather than
            # merely descriptive. Costs one Linear(gate_in, 1) over per_user.
            self.fusion_lambda = nn.Embedding(self.n_users, 1)
            self.fusion_lambda_ctx = nn.Linear(gate_in, 1)
        elif self.fusion_mode not in ("convex", "additive"):
            raise NotImplementedError(
                "fusion_mode must be convex | additive | per_user | hybrid")

        # ablation helpers (cheap; created always, used only when selected)
        self.static_gate = nn.Parameter(torch.zeros(1))      # sigmoid(0)=0.5
        self.concat_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        if self.ablation == "np_vector_gate":
            # zeros init consumes no RNG, so every other condition is unchanged
            self.vector_gate = nn.Parameter(torch.zeros(self.hidden_size))

        self.initializer_range = config["initializer_range"]
        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            raise NotImplementedError("loss_type must be 'CE' or 'BPR'")

        self.apply(self._init_weights)
        self._maybe_load_profiles(config["llm_profile_path"])

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
        """Per-request context vector x_ctx [B, n_context_features(+1)]."""
        if self.context_fields and all(f in interaction for f in self.context_fields):
            cols = [interaction[f].float().unsqueeze(-1) for f in self.context_fields]
            ctx = torch.cat(cols, dim=-1)
        else:
            # fallback: only session length is always available
            b = item_seq_len.size(0)
            ctx = torch.zeros(b, self.n_context_features, device=item_seq_len.device)
            ctx[:, 0] = item_seq_len.float() / self.max_seq_length
        if self._append_hist:
            if self.hg_mode == "seqlen":
                # saturates at the seq cap
                h = (item_seq_len.float() / self.max_seq_length).clamp(max=1.0)
            else:  # logfull: unsaturated total-history signal in [0, 1]
                h = self._hist_log(interaction[self.USER_ID])
            ctx = torch.cat([ctx, h.unsqueeze(-1)], dim=-1)
        return ctx

    def _hist_log(self, user):
        """Normalised log user-activity level in [0, 1] (logfull/suppress)."""
        return torch.log1p(self.user_hist_len[user]) / self.log_hist_norm

    def _gate(self, context, ref):
        if self.ablation == "static_gate":
            return torch.sigmoid(self.static_gate).expand(ref.size(0), self.hidden_size)
        if self.ablation == "np_vector_gate":
            return torch.sigmoid(self.vector_gate).expand(ref.size(0), self.hidden_size)
        if self.ablation == "np_shuffled_ctx":
            context = context[torch.randperm(context.size(0), device=context.device)]
        return self.gating_mlp(context)                      # g [B, H]

    # ----- fused representation -------------------------------------------
    def _fuse(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        z_short = self._short_term(item_seq, item_seq_len)
        z_long = self._long_term(interaction[self.USER_ID])
        if self.ablation in ("no_profiler", "np_vector_gate", "np_shuffled_ctx"):
            z_long = torch.zeros_like(z_long)
        if self._suppress:
            # damp the profile for thin histories (learnt threshold tau/softness beta)
            h = self._hist_log(interaction[self.USER_ID])
            s = torch.sigmoid((h - self.hg_tau) / self.hg_beta.abs().clamp(min=1e-3))
            z_long = z_long * s.unsqueeze(-1)
        if self.ablation == "concat":
            return self.concat_proj(torch.cat([z_long, z_short], dim=-1))
        ctx = self._build_context(interaction, item_seq_len)
        g = self._gate(ctx, z_short)
        lam = self._fusion_lambda(interaction[self.USER_ID], ctx)  # [B,1] or scalar
        # lam 0 -> g*z_long + (1-g)*z_short (identical to the incumbent);
        # lam 1 -> z_short + g*z_long
        return z_short + g * (z_long - (1.0 - lam) * z_short)  # z [B, H]

    def _fusion_lambda(self, user, ctx=None):
        """Fusion-form selector in [0,1]; broadcasts against [B, H]."""
        if self.fusion_mode == "convex":
            return 0.0
        if self.fusion_mode == "additive":
            return 1.0
        if self.fusion_mode == "hybrid":
            return torch.sigmoid(
                self.fusion_lambda_ctx(ctx) + self.fusion_lambda(user))
        return torch.sigmoid(self.fusion_lambda(user))       # [B, 1]

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
