"""
eompp.layers
============

Message-passing layers and tensor helpers shared by every model in the
repository.  The layers are task-agnostic: the node-classification and
graph-classification nets (which differ in pooling / residual / dropout /
head) wrap these same layers.

  * GCNLayer / GINLayer  : clique-expansion graph convolutions.
  * AllDeepSetsLayer     : hypergraph-native Deep Sets layer (Chien 2022).
  * EOPatternLayer       : the proposed EO-Pattern layer; two booleans
                           (use_pee, use_chi) select the ablation variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Scatter / MLP helpers.
# --------------------------------------------------------------------------
def scatter_sum(src, index, dim_size):
    """Sum rows of `src` into `dim_size` buckets given by `index`."""

    out = torch.zeros(dim_size, src.size(-1), dtype=src.dtype,
                      device=src.device)
    return out.index_add(0, index, src)


def scatter_mean(src, index, dim_size):
    """Mean of rows of `src` per bucket (used for per-graph pooling)."""

    summed = scatter_sum(src, index, dim_size)
    ones = torch.ones(index.size(0), dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count = count.index_add(0, index, ones).clamp(min=1.0).unsqueeze(-1)
    return summed / count


def mlp(in_dim, hidden, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, out_dim))


# --------------------------------------------------------------------------
# Clique-expansion graph layers.
# --------------------------------------------------------------------------
class GCNLayer(nn.Module):
    """Symmetric-normalised GCN convolution with self loops.

    Update rule (Kipf & Welling, 2017):

        h_v' = W * ( sum_{u in N(v)} 1/sqrt(d~_v * d~_u) * h_u  +  h_v / d~_v )

    where d~_v = d_v + 1 accounts for the added self-loop.
    """

    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, h, edge_index, in_deg):
        src, tgt = edge_index[0], edge_index[1]
        deg = in_deg + 1.0                                   # add self loop
        norm = (deg[src] * deg[tgt]).rsqrt().unsqueeze(-1)   # 1/sqrt(d_u d_v)
        agg = scatter_sum(norm * h[src], tgt, h.size(0))
        agg = agg + h / deg.unsqueeze(-1)                    # self-loop term
        return self.lin(agg)


class GINLayer(nn.Module):
    """Graph Isomorphism Network convolution.

    Update rule:

        h_v' = MLP( (1 + eps) * h_v  +  sum_{u in N(v)} h_u )

    where eps is a learnable scalar.  Aggregation is an unweighted sum,
    giving strictly greater expressive power than GCN under 1-WL.
    """

    def __init__(self, hidden):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = mlp(hidden, hidden, hidden)

    def forward(self, h, edge_index, in_deg):
        agg = scatter_sum(h[edge_index[0]], edge_index[1], h.size(0))
        return self.mlp((1.0 + self.eps) * h + agg)


# --------------------------------------------------------------------------
# Hypergraph-native Deep Sets layer (Chien et al., 2022).
# --------------------------------------------------------------------------
class AllDeepSetsLayer(nn.Module):
    """One AllDeepSets layer: Deep Sets in both propagation directions.

        z_e  = rho_V( mean_{v in e}  phi_V(h_v) )      (node  -> hyperedge)
        h_v' = rho_E( mean_{e in v}  phi_E(z_e) )      (hyperedge -> node)
    """

    def __init__(self, hidden, dropout):
        super().__init__()
        self.phi_v = mlp(hidden, hidden, hidden)
        self.rho_v = mlp(hidden, hidden, hidden)
        self.phi_e = mlp(hidden, hidden, hidden)
        self.rho_e = mlp(hidden, hidden, hidden)
        self.dropout = dropout

    def forward(self, h, inc, inc_t, he_deg, node_deg):
        # node -> hyperedge
        he = torch.sparse.mm(inc_t, self.phi_v(h))           # M x H, summed
        he = he / he_deg.clamp(min=1.0).unsqueeze(-1)        # mean
        z = F.dropout(F.relu(self.rho_v(he)), self.dropout, self.training)
        # hyperedge -> node
        nd = torch.sparse.mm(inc, self.phi_e(z))             # N x H, summed
        nd = nd / node_deg.clamp(min=1.0).unsqueeze(-1)      # mean
        return self.rho_e(nd)


# --------------------------------------------------------------------------
# The proposed EO-Pattern layer.
# --------------------------------------------------------------------------
class EOPatternLayer(nn.Module):
    """One EO-Pattern message passing layer.

        m_v   = sum_{u in N_H(v)}  w_{vu} * psi(h_v, h_u, chi_vu)
        h_v'  = phi(h_v, m_v)

    with  w_{vu} = P^EE(v, u)        if use_pee else  1 / |N_H(v)|
    and   psi conditioned on chi_vu  if use_chi.

    Both P^EE and the uniform mean are convex combinations, so the two
    aggregation modes are on the same scale.  Dropout/residual are applied
    by the wrapping net, not by the layer.
    """

    def __init__(self, hidden, chi_dim, use_pee, use_chi):
        super().__init__()
        self.use_pee = use_pee
        self.use_chi = use_chi
        psi_in = 2 * hidden + (chi_dim if use_chi else 0)
        self.psi = mlp(psi_in, hidden, hidden)
        self.phi = mlp(2 * hidden, hidden, hidden)

    def forward(self, h, edge_index, p_ee, chi, in_deg):
        src, tgt = edge_index[0], edge_index[1]
        parts = [h[tgt], h[src]]
        if self.use_chi:
            parts.append(chi)
        msg = self.psi(torch.cat(parts, dim=-1))

        if self.use_pee:
            weight = p_ee.unsqueeze(-1)                      # rows sum to 1
        else:
            weight = (1.0 / in_deg[tgt]).unsqueeze(-1)       # uniform mean
        msg = weight * msg

        agg = scatter_sum(msg, tgt, h.size(0))
        return self.phi(torch.cat([h, agg], dim=-1))
