# Experiment 3 - Complementarity (P/Q/R synthetic)

A controlled synthetic experiment that isolates the one regime the other two
experiments never reach: a task where **both** hypergraph-derived ingredients
of the EO-Pattern layer are *simultaneously* load-bearing.

## Why this experiment exists

The two other experiments each leave one ingredient inert:

* **Synthetic twin** - with constant node features, any row-stochastic weight
  (uniform mean or `P^EE`) is inert, so only `chi` (consumed by ψ as a feature)
  can discriminate; `P^EE`-only sits at chance. (`P^EE` itself does differ
  between the twins; it is just unusable on featureless nodes.)
* **Real co-citation benchmark** - `chi` is inert (shuffling it does not hurt:
  `EO-E ≈ EO-D`), and no ingredient yields a gain beyond the noise band over the
  EO-A baseline.

So neither experiment validates the *combination*. This one does, by
construction: it builds a task where `P^EE` alone and `chi` alone each lose
part of the relevant information, while their joint signature identifies the
informative neighbour.

## The P/Q/R gadget

Recall the two quantities and what each normalisation destroys:

```
Z[v,u]      = sum_{e in E(v,u)} 1/(|e|-1)      (raw connection mass)
P^EE(v,u)   = Z[v,u] / d_H(v)                  (normalised by degree -> loses absolute cardinality)
chi_vu      = (binned Z) / Z[v,u]              (normalised by Z -> loses magnitude / multiplicity)
```

For every target node `v` we attach three neighbour types:

| neighbour |                  construction                  | Z[v,·] | chi |
|-----------|------------------------------------------------|--------|-----|
|   **P**   |         one size-2 hyperedge `{v, uP}`         |    1   |  e2 |
|   **Q**   | two size-3 hyperedges `{v,uQ,x1}`, `{v,uQ,x2}` |    1   |  e3 |
|   **R**   |       one size-3 hyperedge `{v, uR, x3}`       |   1/2  |  e3 |

* `chi` separates **P** from `{Q, R}` (e2 vs e3) but **not** Q from R - its
  normalisation erases the multiplicity that distinguishes them, so
  `chi_{v,uQ} = chi_{v,uR} = e3`.
* `P^EE` separates **R** from `{P, Q}` (Z = 1/2 vs Z = 1) but **not** P from Q
  - its normalisation erases the absolute cardinality, so `Z_P = Z_Q = 1`.
* Only `(chi, P^EE)` **jointly** single out **Q** = `(e3, Z=1)`.

The Q-neighbour carries the true class of `v` in its feature; the P-type and
R-type neighbours are decoys with random classes; `v` and the size-3 fillers
are neutral. Recovering `y_v` therefore benefits from identifying Q among the
decoys. Within the normalized `(P^EE, chi)` factorization, Q is identifiable
from the joint signature, but not from either quantity alone.

`P` and `R` are *both* required: drop `P` and `chi` becomes unnecessary
(R is then separable by `P^EE` alone); drop `R` and `P^EE` becomes unnecessary
(P is then separable by `chi` alone). The triple is the minimal gadget in which
neither ingredient is redundant.

## Models

The same nine models as the benchmark (from `eompp.node_models`, unchanged):
MLP, Clique-GCN, Clique-GIN, AllDeepSets, and the EO ablations EO-A..E. EO-E is
EO-D evaluated on globally shuffled `chi` - the negative control.

## Results

Test accuracy (%), mean +/- std. Config: `m_decoys = 4`, 450 gadgets
(7 200 nodes), hidden 64, **1 layer**, 10 seeds.

|           Model           |     Accuracy     | Δ over EO-A |
|---------------------------|------------------|-------------|
|            MLP            |   54.2 +/- 2.1   | - (chance)  |
|        Clique-GCN         |   63.5 +/- 3.8   |    floor    |
|        Clique-GIN         |   63.3 +/- 3.5   |    floor    |
|        AllDeepSets        |   54.9 +/- 0.0   | - (chance)  |
| EO-A: uniform EO baseline |   62.5 +/- 3.9   |     0.0     |
|      EO-B: P^EE only      |   68.1 +/- 3.2   |  **+5.6**   |
|      EO-C: chi only       |   64.2 +/- 4.6   |  **+1.7**   |
|   **EO-D: EO-Pattern**    | **79.5 +/- 3.0** |  **+17.0**  |
|    EO-E: shuffled chi     |   68.6 +/- 3.5   |  (≈ EO-B)   |

### Reading the results

* **The interaction effect is the headline.** `P^EE` alone adds +5.6; `chi`
  alone adds only marginally (+1.7); together they add **+17.0**. The descriptor
  is substantially more useful in conjunction with `P^EE` than as a standalone
  ingredient. This is exactly the regime no other experiment reaches.
* **The EO-E control behaves correctly.** Shuffling `chi` collapses EO-D back
  to the `P^EE`-only level (68.6 ≈ EO-B 68.1). The +17.0 depends on the correct
  structural assignment of `chi`, not merely on increasing the message input
  dimension; this is what the shuffled-`chi` control tests.
* **Dual of the real benchmark.** In co-citation `chi` was inert even in
  combination; here it is inert *alone* but decisive *in combination*. The two
  results are consistent: `chi`'s value is conditional on the structure carrying
  the right signal, which the gadget supplies and co-citation does not.

### Honest caveat

The uniform EO floor sits at ~63 %, not at chance, because the Q-neighbour
always carries the true class, so a structure-blind average recovers a weak but
consistent component of it. Raising `--m-decoys` lowers this floor (`m=2` → ~73 %,
`m=4` → ~63 %, higher → closer to chance) without changing the EO-D gap. A
provable chance-level floor (as in the twin experiment) would require removing
the feature-content signal entirely; this adds complexity without strengthening
the argument. What matters is the ~17-point EO-D gap over the floor and the EO-E
collapse.

This experiment demonstrates that the combination can be genuinely useful
within the normalized `(P^EE, chi)` decomposition; it does not claim the
combination is necessary in general or on real data: the benchmark already shows
it is not, for co-citation.

## Files

|                File                |                                     Purpose                                      |
|------------------------------------|----------------------------------------------------------------------------------|
|              `pqr.py`              | P/Q/R generator + target-only split + driver (the only experiment-specific code) |
|   `results_complementarity.json`   |                        results (per-seed scores included)                        |
| `requirements_complementarity.txt` |                    pure PyTorch + NumPy + SciPy (no TopoNetX)                    |

The EO quantities, the nine models and the train/eval routines all come from
the shared `eompp` package - see the root README. This folder no longer carries
copies of `data_*` / `models_*` / `experiment_*`.

## How to run

```bash
pip install -e ..                              # installs the eompp package

# canonical configuration (reproduces the table above)
python pqr.py --n-gadgets 450 --m-decoys 4 --n-seeds 10 --n-layers 1

# lower the clique floor / sharpen the contrast
python pqr.py --m-decoys 6

# inspect the gadget signatures without training
python -c "from pqr import make_pqr_dataset, check_signatures; \
check_signatures(make_pqr_dataset(n_gadgets=5))"
```

Each seed controls two independent sources of randomness: the train/val/test
split of the target nodes and the model weight initialisation. The hypergraph
structure itself is fixed (`seed=12345` in `make_pqr_dataset`) and is identical
across all seeds - only the split and the initialisation vary between runs.

`check_signatures` prints the `(cardinality, Z)` signatures of the first
target's neighbourhood; with `m_decoys = k` it should report
`{(2, 1.0): k, (3, 1.0): 1, (3, 0.5): 2k+2}` - confirming P = (e2, 1),
Q = (e3, 1), R = (e3, 1/2), with the extra `(3, 0.5)` entries being the neutral
size-3 fillers, which share R's signature and only dilute.
