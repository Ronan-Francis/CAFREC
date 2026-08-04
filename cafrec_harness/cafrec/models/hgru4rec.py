"""HGRU4Rec — Hierarchical RNN baseline (Quadrana et al., RecSys 2017).

RecBole ships no HGRU4Rec/HRNN (its `hrm` is a different model), so this is a
custom `SequentialRecommender`, registered exactly like CAFREC. It supplies the
RNN baseline the plan calls for (RON-10), sitting alongside SASRec and HGN.

Architecture
------------
Two stacked GRUs give the hierarchy:

  * **Session-GRU** runs over the item embeddings of the recent-history window
    and produces a session representation `s` (its last valid hidden state).
  * **User-GRU** maintains a per-user state `h_usr`. That state (a) is projected
    to initialise the session-GRU hidden state — so the user's longer-term
    interest conditions the within-session dynamics (the key HGRU idea) — and
    (b) is updated by the session representation to produce the final
    user-aware sequence representation used for scoring.

RecBole adaptation (documented deviation)
-----------------------------------------
The original HGRU4Rec carries the user-GRU hidden state ACROSS a user's real
session boundaries. RecBole's leave-one-out sequential loader hands the model a
single flattened recent-item window per target (`item_id_list`, up to
`MAX_ITEM_LIST_LENGTH`) with no in-window session delimiters, so true
cross-session propagation is not expressible in one forward pass here. We
therefore approximate the cross-session term with a **learnable per-user initial
state** (`user_state`) that seeds the user-GRU and, through it, the session-GRU.
This preserves the hierarchical structure and the user->session conditioning
while staying within RecBole's batching. If a per-interaction session-index list
is later materialised into the atomic file and surfaced in the batch, the window
can be segmented and the user-GRU stepped per sub-session for the faithful
variant; that wiring is intentionally out of scope here.

The scoring head (dot product with the shared item embedding) and the CE/BPR
loss switch mirror `cafrec.py` so the training contract is identical to the
other models in the registry.
"""
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class HGRU4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.n_users = dataset.user_num
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.loss_type = config["loss_type"]

        # shared item embedding (also the scoring head), padding row = 0
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        # learnable per-user initial state (the cross-session approximation)
        self.user_state = nn.Embedding(self.n_users, self.hidden_size)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        # session-level GRU over the item window
        self.session_gru = nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True, bias=False,
        )
        # user-level GRU: one step, consumes the session representation
        self.user_gru = nn.GRU(
            input_size=self.hidden_size, hidden_size=self.hidden_size,
            num_layers=self.num_layers, batch_first=True, bias=False,
        )
        # project the user state to seed the session-GRU hidden state
        self.user_to_session = nn.Linear(self.hidden_size, self.hidden_size)

        if self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        else:
            raise NotImplementedError("loss_type must be 'CE' or 'BPR'")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)

    def _seq_repr(self, interaction):
        """User-aware sequence representation z [B, H] for the batch."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user = interaction[self.USER_ID]

        emb = self.emb_dropout(self.item_embedding(item_seq))     # [B, L, H]

        # user state seeds the session-GRU hidden state (user -> session)
        h_usr = self.user_state(user)                             # [B, H]
        h0 = torch.tanh(self.user_to_session(h_usr))              # [B, H]
        h0 = h0.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()  # [n_layers, B, H]

        ses_out, _ = self.session_gru(emb, h0)                    # [B, L, H]
        # session representation = hidden at the last VALID position
        s = self.gather_indexes(ses_out, item_seq_len - 1)        # [B, H]

        # user-GRU consumes the session repr, seeded by the user state (session -> user)
        h_usr0 = h_usr.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        usr_out, _ = self.user_gru(s.unsqueeze(1), h_usr0)        # [B, 1, H]
        return usr_out.squeeze(1)                                 # z [B, H]

    def calculate_loss(self, interaction):
        seq_output = self._seq_repr(interaction)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "CE":
            logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
            return self.loss_fct(logits, pos_items)
        neg_items = interaction[self.NEG_ITEM_ID]
        pos_score = (seq_output * self.item_embedding(pos_items)).sum(dim=-1)
        neg_score = (seq_output * self.item_embedding(neg_items)).sum(dim=-1)
        return self.loss_fct(pos_score, neg_score)

    def predict(self, interaction):
        seq_output = self._seq_repr(interaction)
        test_item_emb = self.item_embedding(interaction[self.ITEM_ID])
        return (seq_output * test_item_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        seq_output = self._seq_repr(interaction)
        return torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
