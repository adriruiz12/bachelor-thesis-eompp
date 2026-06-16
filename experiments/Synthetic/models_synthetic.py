"""
models_synthetic.py
==================

Graph-classification models for the synthetic twin-pair experiment.

All models share the same shape:

    x  --(input projection)-->  L message-passing layers  -->
    mean pooling over nodes  -->  MLP classifier  -->  2 logits

The message-passing layers (GCN, GIN, EO-Pattern) are imported from the
shared `eompp.layers`; this module only defines the graph-classification
wrappers (pooling + MLP head) and the ablation registry.  Unlike the
node-classification nets in `eompp.node_models`, these use no dropout and
no residual: the synthetic graphs are tiny and the task is separable, so
the simplest wrapper keeps the 50%/100% result clean.
"""

import torch.nn as nn
import torch.nn.functional as F

from eompp.layers import scatter_mean, mlp, GCNLayer, GINLayer, EOPatternLayer


# --------------------------------------------------------------------------
# Graph baselines on the clique expansion.
# --------------------------------------------------------------------------
class CliqueGNN(nn.Module):
    """GCN or GIN stack + per-graph mean pooling + MLP head.

    Operates solely on the pairwise graph induced by the hyperedges;
    hyperedge cardinalities are invisible to this model.
    """

    def __init__(self, feat_dim, hidden=64, n_layers=2, n_classes=2, kind="gcn"):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden)
        layer_cls = GCNLayer if kind == "gcn" else GINLayer
        self.layers = nn.ModuleList(layer_cls(hidden) for _ in range(n_layers))
        self.head = mlp(hidden, hidden, n_classes)

    def forward(self, batch):
        h = F.relu(self.input_proj(batch["x"]))
        for layer in self.layers:
            h = F.relu(layer(h, batch["edge_index"], batch["in_deg"]))
        graph = scatter_mean(h, batch["batch"], batch["num_graphs"])
        return self.head(graph)


# --------------------------------------------------------------------------
# The proposed hypergraph-derived model.
# --------------------------------------------------------------------------
class EOPatternNet(nn.Module):
    """EO-Pattern layer stack + per-graph mean pooling + MLP head."""

    def __init__(self, feat_dim, chi_dim, hidden=64, n_layers=2,
                 n_classes=2, use_pee=True, use_chi=True):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden)
        self.layers = nn.ModuleList(
            EOPatternLayer(hidden, chi_dim, use_pee, use_chi)
            for _ in range(n_layers))
        self.head = mlp(hidden, hidden, n_classes)

    def forward(self, batch):
        h = F.relu(self.input_proj(batch["x"]))
        for layer in self.layers:
            h = F.relu(layer(h, batch["edge_index"], batch["p_ee"],
                             batch["chi"], batch["in_deg"]))
        graph = scatter_mean(h, batch["batch"], batch["num_graphs"])
        return self.head(graph)


# --------------------------------------------------------------------------
# Registry.
# --------------------------------------------------------------------------
def build_model(name, feat_dim, chi_dim, hidden=64, n_layers=2):
    """Instantiate a model by name (see MODEL_SPECS)."""
    if name == "Clique-GCN":
        return CliqueGNN(feat_dim, hidden, n_layers, kind="gcn")
    if name == "Clique-GIN":
        return CliqueGNN(feat_dim, hidden, n_layers, kind="gin")
    eo = dict(feat_dim=feat_dim, chi_dim=chi_dim,
              hidden=hidden, n_layers=n_layers)
    if name == "A: uniform EO baseline":
        return EOPatternNet(use_pee=False, use_chi=False, **eo)
    if name == "B: EO weights only":
        return EOPatternNet(use_pee=True, use_chi=False, **eo)
    if name == "C: chi only":
        return EOPatternNet(use_pee=False, use_chi=True, **eo)
    if name == "D: EO-Pattern (full)":
        return EOPatternNet(use_pee=True, use_chi=True, **eo)
    if name == "E: shuffled chi":
        return EOPatternNet(use_pee=True, use_chi=True, **eo)
    raise ValueError(f"unknown model name: {name}")


MODEL_SPECS = [
    ("Clique-GCN",             False),
    ("Clique-GIN",             False),
    ("A: uniform EO baseline", False),
    ("B: EO weights only",     False),
    ("C: chi only",            False),
    ("D: EO-Pattern (full)",   False),
    ("E: shuffled chi",        True),
]
