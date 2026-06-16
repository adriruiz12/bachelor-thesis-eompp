# Experiment 2 - Realistic benchmark (EO-Pattern MPNN)

Node classification on the standard **co-citation hypergraph** benchmarks
(Cora, Citeseer). This experiment complements the synthetic one: the
synthetic experiment isolates *what* the EO-Pattern model can do that
clique-expansion models provably cannot; this one measures whether that
ability translates into accuracy on real datasets with real features.

## What the experiment does

For every dataset, every model is trained from scratch and evaluated on a
held-out test set. Nine models are compared:

|    Group     |     Model     |              Structure used              |
|--------------|---------------|------------------------------------------|
|   control    |     `MLP`     |           none (features only)           |
|    graph     | `Clique-GCN`  |    clique expansion of the hypergraph    |
|    graph     | `Clique-GIN`  |    clique expansion of the hypergraph    |
|  hg-derived  | `AllDeepSets` |  incidence matrix (Chien et al., 2022)   | (Note: our reimpl. of it)
|   ablation   |    `EO-A`     | EO layer, no P^EE, no chi (uniform mean) |
|   ablation   |    `EO-B`     |           EO layer, P^EE only            |
|   ablation   |    `EO-C`     |            EO layer, chi only            |
| **proposed** |    `EO-D`     |      **EO layer, full: P^EE + chi**      |
|   control    |    `EO-E`     |     EO-D with chi globally shuffled      |

`EO-A..E` is the ablation study: it isolates the contribution of the two
hypergraph-derived ingredients (the equal-edges random-walk kernel `P^EE` and
the cardinality descriptor `chi`). `EO-E` is a negative control - if shuffled
chi did as well as real chi, chi would be carrying no signal.

## Data source

The datasets are committed in `data/` for full reproducibility. See
`data/README.md` for provenance (HyperGCN commit hash + SHA-256 checksums).

|          | nodes  | features | classes | hyperedges |
|----------|--------|----------|---------|------------|
|   Cora   |  2 708 |  1 433   |    7    |   1 579    |
| Citeseer |  3 312 |  3 703   |    6    |   1 079    |

## Where TopoX is used

The hypergraph is built with **TopoNetX** (`ColoredHyperGraph`), the package
from the TopoX suite. `data_benchmark.py` adds each hyperedge as a rank-1 cell
and reads back the node–hyperedge incidence matrix B via `incidence_matrix(0, 1)`.
TopoNetX orders the rows of B by node-insertion order, so the loader reorders
them to the canonical `0..N-1` indexing. The dependency-free scipy builder from
`eompp.eo` is kept as a fallback and cross-check (the two incidence matrices are
asserted to agree). Everything EO-specific (`P^EE`, `chi`) is then computed from
B by `eompp.eo.eo_quantities_sparse`.

## Files

|                     File                     |                                 Purpose                                 |
|----------------------------------------------|-------------------------------------------------------------------------|
|                   `data/`                    |           committed datasets + provenance (`data/README.md`)            |
|             `data_benchmark.py`              | data loading, TopoNetX hypergraph construction, split, top-level loader |
|          `experiment_benchmark.py`           |                 config + multi-seed driver (resumable)                  |
| `results_cora.json`, `results_citeseer.json` |             per-dataset results (per-seed scores included)              |
|         `requirements_benchmark.txt`         |                      dependencies (incl. TopoNetX)                      |

The nine models and the train/eval routines live in the shared `eompp`
package (`eompp.node_models`, `eompp.node_training`) - see the root README.

## How to run

```bash
pip install -e ..                              # installs the eompp package
pip install -r requirements_benchmark.txt      # adds TopoNetX

# full benchmark, both datasets
python experiment_benchmark.py

# a single dataset
python experiment_benchmark.py --dataset cora

# fast smoke test
python experiment_benchmark.py --quick
```

The driver supports incremental / resumable runs: `--models` restricts the run
to a subset of models, `--out` merges into an existing results file, and
progress is saved after **every seed** - a killed run never loses completed
work.

## Results

Test accuracy (%), mean +/- std over 20 seeds. *all*: accuracy over all
nodes. *active*: accuracy restricted to nodes that belong to at least one
hyperedge (53% of Cora, 44% of Citeseer). All structural node-classification
architectures, except the feature-only MLP control, share the same per-layer
residual connection (`h <- relu(out) + h`, He et al. 2016), 2 layers, hidden
width 128.

|           Model            |   Cora (all)   | Cora (active)  | Citeseer (all) | Citeseer (active) |
|----------------------------|----------------|----------------|----------------|-------------------|
|            MLP             |  73.8 +/- 1.4  |  75.1 +/- 2.0  |  72.2 +/- 0.9  |    74.9 +/- 1.9   |
|         Clique-GCN         |  79.8 +/- 1.7  |  84.8 +/- 1.7  |  73.3 +/- 1.3  |    76.1 +/- 2.0   |
|         Clique-GIN         |  76.5 +/- 1.4  |  82.8 +/- 1.5  |  71.3 +/- 1.0  |    73.9 +/- 2.0   |
|        AllDeepSets         |  75.2 +/- 1.5  |  76.5 +/- 2.0  |  73.1 +/- 1.0  |    75.9 +/- 1.9   |
| EO-A: Uniform EO baseline  |  76.2 +/- 1.2  |  80.3 +/- 1.4  |  72.4 +/- 1.1  |    75.7 +/- 1.6   |
|      EO-B: P^EE only       |  76.6 +/- 1.3  |  80.9 +/- 1.6  |  72.5 +/- 1.1  |    76.0 +/- 1.5   |
|       EO-C: chi only       |  76.2 +/- 1.5  |  80.4 +/- 2.2  |  72.6 +/- 1.1  |    76.0 +/- 1.8   |
|    **EO-D: EO-Pattern**    |  76.3 +/- 1.4  |  80.8 +/- 1.8  |  72.5 +/- 0.9  |    76.2 +/- 1.6   |
|     EO-E: shuffled chi     |  76.6 +/- 1.3  |  80.8 +/- 1.9  |  72.6 +/- 1.1  |    76.1 +/- 1.6   |

### Reading the results

This is a **negative result for `chi` on co-citation**, and it is the honest
headline. On both datasets `EO-E ≈ EO-D`, so the specific `chi`-to-edge
assignment carries no usable signal here. On Cora the ablation ladder
EO-A..EO-D spans 76.2→76.6 with per-model std ≈ 1.2–1.5; all differences lie
within one standard deviation. The ~2.4-pt gain of the EO variants over the
MLP (73.8) is consistent across all 20 seeds and is attributable to
neighbourhood aggregation already present in EO-A; neither `P^EE` nor `chi`
add anything on top of it. On Citeseer all EO variants are indistinguishable
from the MLP.

**Clique-GCN with the same residual connection is clearly stronger on Cora
(79.8/84.8) and matches the EO variants within noise on Citeseer
(73.3/76.1).**

A third caveat: 47% of Cora nodes and 56% of Citeseer nodes lie in no
hyperedge of size ≥2 and receive no propagation under any model — this
affects all models' *all*-node numbers similarly and does not differentially
favour EO or the explicit clique-expansion baselines.

The synthetic twin experiment remains the clean proof that `chi` *can* carry
structure clique expansion destroys, and the P/Q/R gadget shows that the joint
signature of `P^EE` and `chi` can be useful within the normalized factorization;
co-citation simply is not a structure where either helps.

## Note on compute

Both datasets were run on CPU (hidden 128, 2 layers, **20 seeds**, lr = 1e-3),
following the AllSet evaluation protocol (Chien et al. 2022).

To reproduce:

```bash
python experiment_benchmark.py --dataset cora     --n-seeds 20 --out results_cora.json
python experiment_benchmark.py --dataset citeseer --n-seeds 20 --out results_citeseer.json
```
