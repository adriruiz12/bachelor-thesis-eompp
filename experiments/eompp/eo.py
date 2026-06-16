"""
eompp.eo
========

Hypergraph helpers and the EO-Pattern structural quantities shared by all
three experiments.  This is the single source of truth for the hypergraph-
derived quantities used below:

  * P^EE(v, u) : equal-edges random-walk transition kernel used in the
        Eidi-Otter-inspired construction.
        With w_e = 1/(|e|-1) and Z = B diag(w) B^T,
            P^EE(v, u) = Z[v, u] / d_H(v)            (v != u)
        the rows of P^EE sum to 1 by construction.

  * chi_vu     : normalized, 1/(|e|-1)-weighted one-hot histogram of the
        cardinalities of the hyperedges shared by v and u.

`eo_quantities_sparse` works from an incidence matrix B so it scales to
large datasets; the synthetic experiment builds B per small graph with
`build_incidence_scipy` and calls the same function, so there is exactly
one implementation of the formula in the repository.
"""

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------
# Incidence matrix from a hyperedge list (dependency-free).
# --------------------------------------------------------------------------
def build_incidence_scipy(n_nodes, hyperedges):
    """Build the N x M incidence matrix B from a list of hyperedges.

    Hyperedges of size < 2 are dropped (they do not contribute to the walk).
    """

    rows, cols = [], []
    col = 0
    for he in hyperedges:
        he = sorted(set(int(v) for v in he))
        if len(he) >= 2:
            for v in he:
                rows.append(v)
                cols.append(col)
            col += 1
    data = np.ones(len(rows), dtype=np.float64)
    return sp.coo_matrix((data, (rows, cols)),
                         shape=(n_nodes, col)).tocsr()


# --------------------------------------------------------------------------
# Clique expansion (used to verify the synthetic twin pairs).
# --------------------------------------------------------------------------
def clique_expansion(hyperedges, n_nodes=None):
    """Edge set (as a set of frozensets) of the pairwise graph induced by H."""

    edges = set()
    for e in hyperedges:
        e = tuple(sorted(set(int(v) for v in e)))
        if len(e) < 2:
            continue
        for i in range(len(e)):
            for j in range(i + 1, len(e)):
                edges.add(frozenset((e[i], e[j])))
    return edges


# --------------------------------------------------------------------------
# EO-Pattern quantities from the incidence matrix.
# --------------------------------------------------------------------------
def eo_quantities_sparse(B, max_card):
    """Compute edge_index, P^EE and chi from an incidence matrix.

    Parameters
    ----------
    B : scipy sparse, shape [N, M]
        Incidence matrix, rows aligned to node ids 0..N-1.
    max_card : int
        Cardinalities >= max_card are folded into one overflow bin.

    Returns
    -------
    edge_index : int array [2, E]   row 0 = source u, row 1 = target v
    p_ee       : float array [E]    P^EE(v, u) for the target v
    chi        : float array [E, max_card-1]   cardinality descriptor
    d_h        : float array [N]    hypergraph degree
    """

    B = B.tocsr().astype(np.float64)
    card = np.asarray(B.sum(axis=0)).ravel()       # cardinality per hyperedge
    keep = card >= 2
    B = B[:, keep]
    card = card[keep]

    w = 1.0 / (card - 1.0)                         # 1/(|e|-1)
    d_h = np.asarray(B.sum(axis=1)).ravel()        # hypergraph degree
    d_h_safe = np.where(d_h > 0, d_h, 1.0)

    # Z[v,u] = sum_{e in E(v,u)} 1/(|e|-1)
    S = (B @ sp.diags(w) @ B.T).tocoo()
    off = S.row != S.col                           # drop the diagonal
    src = S.col[off]                               # u
    tgt = S.row[off]                               # v
    z_vu = S.data[off]                             # Z_vu

    p_ee = z_vu / d_h_safe[tgt]                    # rows (fixed v) sum to 1

    # chi: one sparse product per cardinality bin.
    n_bins = max_card - 1
    bin_idx = np.minimum(card.astype(int), max_card) - 2
    chi = np.zeros((len(src), n_bins), dtype=np.float64)
    for c in range(n_bins):
        sel = (bin_idx == c).astype(np.float64)
        if sel.sum() == 0:
            continue
        Sc = (B @ sp.diags(w * sel) @ B.T).tocsr()
        chi[:, c] = np.asarray(Sc[tgt, src]).ravel()
    chi /= z_vu[:, None]

    edge_index = np.vstack([src, tgt]).astype(np.int64)
    return edge_index, p_ee, chi, d_h
