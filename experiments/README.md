# EO-Pattern MPNN - experiments

Code for the experiments on the **EO-Pattern** message passing layer
(equal-edges kernel `P^EE` used in the Eidi–Otter-inspired construction + cardinality descriptor `χ`).

Three experiments, each answering a distinct question:

|       Folder       |       Experiment        |                                 Question                                  |         Task         |
|--------------------|-------------------------|---------------------------------------------------------------------------|----------------------|
|    `Synthetic/`    |     1 - Separation      | Does the EO layer detect structure a clique expansion provably destroys?  | graph classification |
|    `Benchmark/`    | 2 - Realistic benchmark |    Does that ability translate into accuracy on real co-citation data?    | node classification  |
| `Complementarity/` |   3 - Complementarity   | Is there a regime where `P^EE` and `χ` provide complementary information? | node classification  |

## Shared core: the `eompp` package

The method lives in the installable `eompp/` package; the three experiments
share a single implementation of the EO formula and the message-passing layers:

|         Module          |                                            Contents                                             |
|-------------------------|-------------------------------------------------------------------------------------------------|
|      `eompp/eo.py`      | `eo_quantities_sparse` (the one EO implementation), `clique_expansion`, `build_incidence_scipy` |
|    `eompp/layers.py`    |     `scatter_sum/mean`, `mlp`, `GCNLayer`, `GINLayer`, `AllDeepSetsLayer`, `EOPatternLayer`     |
|   `eompp/metrics.py`    |                                           `macro_f1`                                            |
| `eompp/node_models.py`  |               the nine node-classification models + `build_model` + `MODEL_SPECS`               |
| `eompp/node_training.py`|                      `to_tensors`, `train_one`, `evaluate`, `shuffled_chi`                      |

The training harness is task-specific and not shared: graph classification
(Synthetic) uses disjoint-union batching with per-graph pooling; node
classification (Benchmark, Complementarity) uses a masked single graph.
Both node-classification experiments share `eompp.node_*`; the
graph-classification harness lives in `Synthetic/`.

## Install

From this directory, once:

```bash
pip install -e .
```

This installs `eompp` (and its dependencies: numpy, scipy, torch) in editable
mode, so every experiment folder can `import eompp`. The Benchmark experiment
additionally needs TopoNetX:

```bash
pip install -r Benchmark/requirements_benchmark.txt
```

## Run

**Experiment 1 - Synthetic (graph classification).** The 50 % results of the clique-only models and chi-free EO variants follow from the paired construction and feature symmetry; the 100 % results of the chi-based variants are empirical outcomes of the committed run.
```bash
cd Synthetic
python experiment_synthetic.py
# variants:  --quick  |  --decomposition triangles  |  --feature-mode random
```

**Experiment 2 - Benchmark (node classification).** 50/25/25 random splits, multi-seed, resumable.
```bash
cd Benchmark
python experiment_benchmark.py --dataset cora     --n-seeds 20 --out reproduced_results_cora.json
python experiment_benchmark.py --dataset citeseer --n-seeds 20 --out reproduced_results_citeseer.json
# quick smoke test:  python experiment_benchmark.py --dataset cora --quick
```

**Experiment 3 - Complementarity (P/Q/R gadget).** Canonical configuration:
```bash
cd Complementarity
python pqr.py --n-gadgets 450 --m-decoys 4 --n-seeds 10 --n-layers 1 --out reproduced_results_complementarity.json
```

Each folder has its own README with the experiment design, the results table,
and per-experiment flags.

## Repository structure

```
.
├── eompp/                          shared method (installable package)
│   ├── eo.py                       EO quantities: P^EE, χ, incidence helpers
│   ├── layers.py                   GCNLayer, GINLayer, AllDeepSetsLayer, EOPatternLayer
│   ├── node_models.py              nine node-classification models + registry
│   ├── node_training.py            train/eval loop, tensor conversion, chi shuffle
│   └── metrics.py                  macro_f1
├── pyproject.toml
├── Synthetic/                      Experiment 1 - twin-pair separation
│   ├── data_synthetic.py           twin-pair generator and split
│   ├── models_synthetic.py         graph-classification wrappers (pooling + head)
│   └── experiment_synthetic.py     training harness and results table
├── Benchmark/                      Experiment 2 - co-citation node classification
│   ├── data/                       committed datasets (Cora, Citeseer)
│   ├── data_benchmark.py           data loading, TopoNetX incidence, split
│   └── experiment_benchmark.py     multi-seed driver (resumable)
└── Complementarity/                Experiment 3 - P/Q/R gadget
    └── pqr.py                      gadget generator, split, and driver
```