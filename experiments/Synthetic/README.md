# Experiment 1 - Synthetic separation (EO-Pattern MPNN)

Synthetic experiment for the EO-Pattern message passing layer
(equal-edges kernel `P^EE` used in the Eidi–Otter-inspired construction +
cardinality descriptor χ).

It answers one question, in a fully controlled setting:

> **Does the EO-Pattern layer detect hyperedge structure that a clique
> expansion provably destroys?**

## The idea

Every example is a *graph-classification* instance. For each random
**base graph** `G` (built as a union of cliques) we produce **two
hypergraphs on the same vertex set**:

|    class    | hypergraph |                                   hyperedges                                   |
|-------------|------------|--------------------------------------------------------------------------------|
|  0 - "big"  |  `H_big`   |                    the original cliques, one hyperedge each                    |
| 1 - "small" |  `H_small` | each clique replaced by smaller hyperedges whose clique expansion is unchanged |

The smaller hyperedges are controlled by `decomposition`:

* `"edges"` (default): each clique `C_i` becomes all its pairwise edges.
* `"triangles"` (harder): each clique `C_i` is covered by triangles through
  one pivot vertex. In the implementation, one pivot vertex is chosen, and
  triangles are formed with the pivot and every pair of the remaining vertices.

**By construction `clique_expansion(H_big) == clique_expansion(H_small)`**
(asserted for every twin pair). Node features are shared between twins. Hence
any model that is a function of the clique expansion alone receives *identical
input* for the two classes and, since the dataset contains both twins,
**cannot exceed 50 % accuracy**. The two hypergraphs differ *only* in
hyperedge cardinality, which is exactly what χ encodes. Splitting is by base
graph, so the 50 %-bound holds inside every split.

## The models (the ablation grid)

Seven models (graph-classification wrappers around the shared `eompp` layers):

|          name          |    weights   |      message input       |                              role                              |
|------------------------|--------------|--------------------------|----------------------------------------------------------------|
|       Clique-GCN       |   GCN norm   |          `h_u`           |               graph baseline on clique expansion               |
|       Clique-GIN       |   GIN sum    |          `h_u`           |               graph baseline on clique expansion               |
| A: uniform EO baseline | uniform mean |        `h_v, h_u`        | same pairwise MPNN skeleton, both hypergraph-derived parts off |
|  B: EO weights only    |    `P^EE`    |        `h_v, h_u`        |                 does the EO kernel alone help?                 |
|       C: χ only        | uniform mean |       `h_v, h_u, χ`      |                does the descriptor alone help?                 |
|  D: EO-Pattern (full)  |    `P^EE`    |       `h_v, h_u, χ`      |                       the proposed layer                       |
|     E: shuffled χ      |    `P^EE`    | `h_v, h_u, χ` (permuted) |                  is χ's signal real structure?                 |

## Results

Test accuracy, mean ± std over 5 seeds, for the committed default run
(`feature_mode=constant`, edge decomposition):

```
          Model           |  Accuracy (%) |  Macro-F1 (%) |
--------------------------|---------------|---------------|
        Clique-GCN        |  50.0 +/- 0.0 | 33.3 +/- 0.0  |
        Clique-GIN        |  50.0 +/- 0.0 | 33.3 +/- 0.0  |
  A: uniform EO baseline  |  50.0 +/- 0.0 | 33.3 +/- 0.0  |
    B: EO weights only    |  50.0 +/- 0.0 | 33.3 +/- 0.0  |
       C: chi only        | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
   D: EO-Pattern (full)   | 100.0 +/- 0.0 | 100.0 +/- 0.0 |
     E: shuffled chi      |  50.0 +/- 0.0 | 33.3 +/- 0.0  |
```

Reading of the table:

* **Clique-GCN / GIN / A** at 50 %: any model that only sees the clique
  expansion is at chance (*by construction*).
* **B at 50 %**: with the default constant node features every node has the
  same input, so any row-stochastic aggregation weight (the uniform mean of A
  *or* `P^EE`) maps constant inputs to constant outputs *regardless of its
  values* - only χ, consumed by ψ as a feature, can break the node symmetry.
  Hence A ≡ B ≡ 50 % by construction.
* **C and D at 100 %**: the cardinality descriptor χ is the only signal
  available to the layer that separates the classes, and it suffices to do so.
* **E at 50 %**: permuting χ across edges destroys the signal, so χ's
  contribution is genuine hypergraph structure, not extra capacity.

## Files

|             file             |                                 content                                 |
|------------------------------|-------------------------------------------------------------------------|
|     `data_synthetic.py`      |            twin-pair dataset generator and base-graph split             |
|    `models_synthetic.py`     | graph-classification wrappers (pooling + MLP head) over `eompp` layers  |
|  `experiment_synthetic.py`   | disjoint-union batching, training, multi-seed evaluation, results table |
|  `results_synthetic.json`    |               results (full config + per-seed accuracies)               |
| `requirements_synthetic.txt` |                              dependencies                               |

The EO quantities (`P^EE`, χ) and the message-passing layers come from the
shared `eompp` package - see the root README. 

## Running

```bash
pip install -e ..                              # installs the eompp package

python experiment_synthetic.py                 # full run (constant features)
python experiment_synthetic.py --quick         # fast sanity check
python experiment_synthetic.py --feature-mode random
python experiment_synthetic.py --decomposition triangles
```

Each run writes a `results_synthetic*.json` with the full config and per-seed
accuracies. Defaults live in the `Config` class in `experiment_synthetic.py`.

## Design notes

* **χ descriptor.** χ_vu is a fixed structural object: a `1/(|e|-1)`-weighted,
  normalised one-hot histogram over the cardinalities of the hyperedges shared
  by `v` and `u`. It is *not* learned (all learning happens in the message
  function ψ that consumes it).
* **Pure PyTorch.** No PyG / DGL. The synthetic graphs are small and a
  self-contained implementation is easier to audit.
* **Why B at chance is the *right* result.** With constant features it is
  *provable* that any row-stochastic weight is inert, so the EO-Pattern signal
  can only enter through χ (the sole ingredient ψ consumes as a feature). That
  is the clean reason - not a cancellation of `P^EE` on the twins, which does
  not hold in general.