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
|         Clique-GCN         |  79.9 +/- 1.5  |  85.0 +/- 1.7  |  73.2 +/- 1.4  |    76.3 +/- 2.1   |
|         Clique-GIN         |  76.5 +/- 1.4  |  82.8 +/- 1.5  |  71.3 +/- 1.0  |    73.9 +/- 2.0   |
|        AllDeepSets         |  75.2 +/- 1.5  |  76.5 +/- 2.0  |  73.1 +/- 1.0  |    75.9 +/- 1.9   |
| EO-A: Uniform EO baseline  |  76.2 +/- 1.2  |  80.3 +/- 1.4  |  72.4 +/- 1.1  |    75.7 +/- 1.6   |
|      EO-B: P^EE only       |  76.6 +/- 1.3  |  80.9 +/- 1.6  |  72.5 +/- 1.1  |    76.0 +/- 1.5   |
|       EO-C: chi only       |  76.2 +/- 1.5  |  80.4 +/- 2.2  |  72.6 +/- 1.1  |    76.0 +/- 1.8   |
|    **EO-D: EO-Pattern**    |  76.3 +/- 1.4  |  80.8 +/- 1.8  |  72.5 +/- 0.9  |    76.2 +/- 1.6   |
|     EO-E: shuffled chi     |  76.6 +/- 1.3  |  80.8 +/- 1.9  |  72.6 +/- 1.1  |    76.1 +/- 1.6   |

### Reading the results

This is a **negative result for `chi` on co-citation**, and it is the honest
headline. On both datasets `EO-E ≈ EO-D`, providing no evidence that the model
uses the structured organization of `chi`. On Cora, variants EO-A..EO-D range
from 76.2 % to 76.6 %. Relative to EO-A, `P^EE` produces only a 0.4-point gain,
while the variants using `chi` provide no clear additional improvement.

The EO variants exceed the feature-only MLP on average, but this difference
cannot be attributed exclusively to neighborhood aggregation because the MLP
is non-residual. Clique-GCN is clearly stronger than the EO variants on Cora.
On Citeseer, the observed accuracies are close and no EO variant provides a
clear advantage.

Finally, 47 % of Cora nodes and 56 % of Citeseer nodes satisfy `d_H(v)=0` and
receive no messages from other nodes under any structural model. The all-node
results therefore mix structurally active nodes with nodes for which
neighborhood propagation is unavailable.

## Note on compute

Both datasets were run on CPU (hidden 128, 2 layers, **20 seeds**, lr = 1e-3),
using independently generated class-stratified 50/25/25 splits.

To reproduce, use a new output filename: if an existing canonical result file is
supplied, the resumable driver will reuse its completed seeds instead of rerunning
them.

```bash
python experiment_benchmark.py --dataset cora     --n-seeds 20 --out reproduced_results_cora.json
python experiment_benchmark.py --dataset citeseer --n-seeds 20 --out reproduced_results_citeseer.json
```
