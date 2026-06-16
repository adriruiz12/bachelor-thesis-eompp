"""
eompp.node_models
=================

The nine node-classification models shared by the Benchmark and
Complementarity experiments.  Every model maps node features X to per-node
class logits:

    X --(input proj)--> L message-passing layers --> linear classifier

  * MLP            : ignores all structure (control baseline).
  * Clique-GCN     : GCN on the clique expansion (graph baseline).
  * Clique-GIN     : GIN on the clique expansion (graph baseline).
  * AllDeepSets    : hypergraph-native incidence baseline (Chien 2022).
  * EOPatternNet   : the proposed model; (use_pee, use_chi) select the
                     ablation variant
                         A: -/-   B: pee/-   C: -/chi   D: pee/chi
                     E is variant D evaluated on permuted chi (handled in
                     eompp.node_training.shuffled_chi).

All models share dropout and a fixed hidden width for a fair comparison.
"""

import torch.nn as nn
import torch.nn.functional as F

from eompp.layers import (GCNLayer, GINLayer, AllDeepSetsLayer,
                          EOPatternLayer)


# --------------------------------------------------------------------------
# Control baseline: a plain MLP.
# --------------------------------------------------------------------------
class MLPNet(nn.Module):
    """Node classifier with no structural information at all."""

    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        dims = [n_features] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, batch):
        h = batch["x"]
        for layer in self.layers:
            h = F.dropout(F.relu(layer(h)), self.dropout, self.training)
        return self.head(h)


# --------------------------------------------------------------------------
# Graph baselines on the clique expansion.
# --------------------------------------------------------------------------
class CliqueGNN(nn.Module):
    """GCN or GIN layer stack on the clique expansion + linear classifier.

    Operates solely on the pairwise graph induced by the hyperedges;
    hyperedge cardinalities are invisible to this model.
    """

    def __init__(self, n_features, hidden, n_classes, n_layers=2,
                 dropout=0.5, kind="gcn"):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        layer_cls = GCNLayer if kind == "gcn" else GINLayer
        self.layers = nn.ModuleList(layer_cls(hidden) for _ in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, batch):
        h = F.dropout(F.relu(self.input_proj(batch["x"])),
                      self.dropout, self.training)
        for layer in self.layers:
           out = F.relu(layer(h, batch["edge_index"], batch["deg"]))
           h = F.dropout(out + h, self.dropout, self.training)   # residual
        return self.head(h)
    

# --------------------------------------------------------------------------
# Hypergraph-native baseline: AllDeepSets.
# --------------------------------------------------------------------------
class AllDeepSetsNet(nn.Module):
    """AllDeepSets layer stack + linear classifier."""

    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.layers = nn.ModuleList(
            AllDeepSetsLayer(hidden, dropout) for _ in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, batch):
        h = F.dropout(F.relu(self.input_proj(batch["x"])),
                      self.dropout, self.training)
        for layer in self.layers:
            out = layer(h, batch["incidence"], batch["incidence_t"],
                        batch["he_deg"], batch["node_deg_hg"])
            h = F.dropout(F.relu(out) + h, self.dropout, self.training)  # residual
        return self.head(h)


# --------------------------------------------------------------------------
# The proposed model: EO-Pattern.
# --------------------------------------------------------------------------
class EOPatternNet(nn.Module):
    """EO-Pattern layer stack + linear classifier."""

    def __init__(self, n_features, chi_dim, hidden, n_classes, n_layers=2,
                 dropout=0.5, use_pee=True, use_chi=True):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.layers = nn.ModuleList(
            EOPatternLayer(hidden, chi_dim, use_pee, use_chi)
            for _ in range(n_layers))
        self.head = nn.Linear(hidden, n_classes)
        self.dropout = dropout

    def forward(self, batch):
        h = F.dropout(F.relu(self.input_proj(batch["x"])),
                      self.dropout, self.training)
        for layer in self.layers:
            out = layer(h, batch["edge_index"], batch["p_ee"],
                        batch["chi"], batch["deg"])
            h = F.dropout(F.relu(out) + h, self.dropout, self.training)  # residual
        return self.head(h)


# --------------------------------------------------------------------------
# Registry.
# --------------------------------------------------------------------------
def build_model(name, n_features, chi_dim, n_classes,
                hidden=128, n_layers=2, dropout=0.5):
    """Instantiate a model by name (see MODEL_SPECS)."""
    common = dict(n_features=n_features, hidden=hidden,
                  n_classes=n_classes, n_layers=n_layers, dropout=dropout)
    if name == "MLP":
        return MLPNet(**common)
    if name == "Clique-GCN":
        return CliqueGNN(kind="gcn", **common)
    if name == "Clique-GIN":
        return CliqueGNN(kind="gin", **common)
    if name == "AllDeepSets":
        return AllDeepSetsNet(**common)
    eo = dict(chi_dim=chi_dim, **common)
    if name == "EO-A: uniform EO baseline":
        return EOPatternNet(use_pee=False, use_chi=False, **eo)
    if name == "EO-B: P^EE only":
        return EOPatternNet(use_pee=True, use_chi=False, **eo)
    if name == "EO-C: chi only":
        return EOPatternNet(use_pee=False, use_chi=True, **eo)
    if name == "EO-D: EO-Pattern (full)":
        return EOPatternNet(use_pee=True, use_chi=True, **eo)
    if name == "EO-E: shuffled chi":
        return EOPatternNet(use_pee=True, use_chi=True, **eo)
    raise ValueError(f"unknown model name: {name}")


# (model name, needs chi shuffling).  First block is the main comparison;
# the EO-* block is the ablation study.
MODEL_SPECS = [
    ("MLP",                       False),
    ("Clique-GCN",                False),
    ("Clique-GIN",                False),
    ("AllDeepSets",               False),
    ("EO-A: uniform EO baseline", False),
    ("EO-B: P^EE only",           False),
    ("EO-C: chi only",            False),
    ("EO-D: EO-Pattern (full)",   False),
    ("EO-E: shuffled chi",        True),
]
