"""
synthetic_data.py
=================

Generator of the synthetic hypergraph-classification dataset.

Design
------
The experiment must show that the EO-Pattern layer detects information
that a clique expansion destroys.  To make that claim airtight we build
the dataset so that **a model that only sees the clique expansion is
provably at chance (50%)**.

For every "base graph" we draw a set of cliques C_1, ..., C_k and define
the graph G as the union of their edge sets.  We then build TWO
hypergraphs on the *same* vertex set:

  * H_big   (label 0, class "big")   : hyperedges = the cliques C_i.
  * H_small (label 1, class "small") : hyperedges = a fine decomposition
                                       of every clique.
        - decomposition="edges"     : every clique becomes its 2-subsets
                                      (all hyperedges have size 2).
        - decomposition="triangles" : every clique is covered by a star
                                      of triangles through one pivot
                                      vertex (all hyperedges have size 3);
                                      requires clique size >= 4.

By construction, clique_expansion(H_big) == clique_expansion(H_small) == G.
(We assert this for every twin pair.)  However, the two hypergraphs have 
different hyperedge structures. The distinguishing signal available to the
EO-Pattern layer enters specifically through the cardinality descriptor
chi, which encodes the sizes of the hyperedges shared by each pair of
vertices.

Node features are shared between the two twins.  Therefore any model that
is a function of the clique expansion alone (GCN, GIN, ...) receives
*identical* input for the two classes of a twin pair, and, since the
dataset contains both twins, cannot exceed 50% accuracy.  The EO-Pattern
layer, in contrast, sees the cardinality descriptor chi and can separate
the classes.

Splitting in train, validation and test sets is done by base graph: both
twins of a base graph always land in the same split, so the 50%-bound
holds within every split.
"""

import numpy as np

from eompp.eo import clique_expansion


# --------------------------------------------------------------------------
# Base graph: a union of cliques.
# --------------------------------------------------------------------------
def generate_base_graph(rng, k_range=(2, 5), size_range=(3, 6), overlap_max=2):
    """Sample a list of cliques (each a sorted list of vertex ids) and n_nodes.

    Cliques are grown one at a time; each new clique may reuse up to
    `overlap_max` vertices from the pool of already-used vertices, which
    creates hypergraphs with non-trivial, irregular structure.
    """
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    cliques = []
    used = 0
    for _ in range(k):
        s = int(rng.integers(size_range[0], size_range[1] + 1))
        if used == 0:
            verts = list(range(0, s))
            used = s
        else:
            o = int(rng.integers(0, min(overlap_max, used, s - 1) + 1))
            reuse = list(rng.choice(used, size=o, replace=False)) if o > 0 else []
            new = list(range(used, used + s - o))
            used += s - o
            verts = [int(v) for v in reuse] + new
        cliques.append(sorted(set(verts)))
    return cliques, used


# --------------------------------------------------------------------------
# Twin hypergraphs from a clique set.
# --------------------------------------------------------------------------
def triangle_star(clique):
    """Cover a clique with triangles through a pivot. Preserves clique exp."""

    c = sorted(clique)
    if len(c) == 3:
        return [tuple(c)]
    pivot = c[0]
    tris = []
    for i in range(1, len(c)):
        for j in range(i + 1, len(c)):
            tris.append(tuple(sorted((pivot, c[i], c[j]))))
    return tris


def cliques_to_hypergraphs(cliques, decomposition="edges"):
    """Return (H_big, H_small) as lists of hyperedges (tuples of vertices)."""

    big = sorted({tuple(sorted(c)) for c in cliques})

    small = set()
    for c in cliques:
        c = sorted(c)
        if decomposition == "edges":
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    small.add((c[i], c[j]))
        elif decomposition == "triangles":
            for t in triangle_star(c):
                small.add(t)
        else:
            raise ValueError(f"unknown decomposition: {decomposition}")
    return list(big), sorted(small)


# --------------------------------------------------------------------------
# Full dataset.
# --------------------------------------------------------------------------
def make_dataset(n_base, seed=0, decomposition="edges",
                 feature_mode="constant", feat_dim=16,
                 k_range=(2, 5), size_range=(3, 6), overlap_max=2):
    """Build the list of labelled examples.

    Each base graph yields two examples (the H_big / H_small twins) that
    share the same node features.  Returns a list of dicts with keys
    `hyperedges`, `n_nodes`, `x`, `label`, `base_id`.
    """

    rng = np.random.default_rng(seed)
    examples = []
    n_degenerate = 0

    for bid in range(n_base):
        cliques, n = generate_base_graph(rng, k_range, size_range, overlap_max)
        big, small = cliques_to_hypergraphs(cliques, decomposition)

        # Drop degenerate base graphs where the two twins coincide
        # (can only happen with decomposition="triangles" and all cliques
        # of size 3).
        if {frozenset(e) for e in big} == {frozenset(e) for e in small}:
            n_degenerate += 1
            # Discards this case
            continue

        # The backbone of the whole argument: verify the twins induce the
        # same pairwise graph.
        assert clique_expansion(big, n) == clique_expansion(small, n), (
            "twin clique expansions differ - the dataset would be unfair")

        # Node features, shared by both twins.
        if feature_mode == "constant":
            x = np.ones((n, feat_dim), dtype=np.float32)
        elif feature_mode == "random":
            x = rng.standard_normal((n, feat_dim)).astype(np.float32)
        else:
            raise ValueError(f"unknown feature_mode: {feature_mode}")

        examples.append(dict(hyperedges=big, n_nodes=n, x=x,
                             label=0, base_id=bid))   # "big"
        examples.append(dict(hyperedges=small, n_nodes=n, x=x,
                             label=1, base_id=bid))   # "small"

    if n_degenerate:
        print(f"[make_dataset] dropped {n_degenerate} degenerate base graphs")
    return examples


def split_dataset(examples, seed=0, fracs=(0.6, 0.2, 0.2)):
    """Split by base graph id so twins never cross a split boundary."""

    base_ids = sorted({e["base_id"] for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(base_ids)
    n = len(base_ids)
    a = int(fracs[0] * n)
    b = int((fracs[0] + fracs[1]) * n)
    train_ids = set(base_ids[:a])
    val_ids = set(base_ids[a:b])
    test_ids = set(base_ids[b:])

    def pick(ids):
        return [e for e in examples if e["base_id"] in ids]

    return pick(train_ids), pick(val_ids), pick(test_ids)


if __name__ == "__main__":
    ex = make_dataset(n_base=20, seed=0, decomposition="edges")
    print(f"examples           : {len(ex)}")
    sizes = [len(e["hyperedges"]) for e in ex]
    nodes = [e["n_nodes"] for e in ex]
    print(f"nodes per graph    : min={min(nodes)} max={max(nodes)}")
    print(f"hyperedges/graph   : min={min(sizes)} max={max(sizes)}")
    print(f"class balance      : "
          f"{sum(e['label'] == 0 for e in ex)} big / "
          f"{sum(e['label'] == 1 for e in ex)} small")
    tr, va, te = split_dataset(ex, seed=0)
    print(f"split (tr/va/te)   : {len(tr)}/{len(va)}/{len(te)}")
