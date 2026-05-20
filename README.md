# eompp

**A Hypergraph-Native Message Passing Layer Based on Eidi-Otter Connectivity Patterns**

This repository contains the research materials for a BSc thesis project ("TFG", as called in Spain) developed at UCLA under the supervision of Prof. Guido Montúfar. The central contribution is a minimal, mathematically grounded message passing layer for hypergraphs defined directly from hyperedge connectivity patterns, without reducing the hypergraph to a graph or relational structure first.

---

## Guiding question

> Can one define message passing on hypergraphs directly from hyperedge connectivity patterns (Eidi & Otter, 2025), rather than by first reducing the hypergraph to a graph or to a general relational structure (Taha et al., ICLR 2025)?

---

## Repository contents

```
eompp/
├── surveys/
│   ├── higher_order_structures.pdf     # Survey of higher-order domains and message passing frameworks
│   └── random_walk_variants.pdf        # Survey of random walk variants and their lifts to higher-order structures
├── proposal/
│   ├── eompp_proposal.pdf              # Technical write-up of the EO-pattern MPNN layer
│   └── eompp_presentation.pdf          # Presentation slides (May 2026)
└── README.md
```

*(Implementation coming next — see Roadmap below.)*

---

## The EO-Pattern MPNN layer

The proposed layer retains the standard MPNN structure (message / aggregate / update) but replaces graph-based propagation with two quantities derived directly from hypergraph incidence:

**Equal-edges transition weight** (Eidi & Otter):

$$P_{\mathrm{EE}}(v, u) = \frac{1}{d_H(v)} \sum_{e \in \mathcal{E}(v,u)} \frac{1}{|e|-1}$$

A neighbour reached through a smaller hyperedge receives more weight than one reached through a larger hyperedge.

**Cardinality-pattern descriptor** (learnable):

$$\chi_{vu} = \frac{1}{Z_{vu}} \sum_{e \in \mathcal{E}(v,u)} \frac{1}{|e|-1}\, \mathbf{a}_{|e|}$$

A weighted average of learnable cardinality embeddings $\mathbf{a}_r \in \mathbb{R}^p$, summarising the sizes of the hyperedges through which $v$ and $u$ interact.

**Layer update:**

$$m_v^{(\ell+1)} = \sum_{u \in \mathcal{N}_H(v)} P_{\mathrm{EE}}(v,u)\, \psi^{(\ell)}\!\left(h_v^{(\ell)},\, h_u^{(\ell)},\, \chi_{vu}\right)$$

$$h_v^{(\ell+1)} = \varphi^{(\ell)}\!\left(h_v^{(\ell)},\, m_v^{(\ell+1)}\right)$$

### Key properties

- **Permutation equivariant** by construction.
- **Distinguishes beyond clique expansion**: e.g., $H_1 = \{\{a,b,c\}\}$ and $H_2 = \{\{a,b\},\{a,c\},\{b,c\}\}$ induce the same triangle graph but different descriptors ($\chi_{ab}^{H_1} = \mathbf{a}_3 \neq \mathbf{a}_2 = \chi_{ab}^{H_2}$).
- **Computational cost**: $O\!\left(\sum_{e \in E} |e|^2\right)$ — same asymptotic cost as clique expansion, but without its semantic loss.

---

## Relation to existing work

| Method | Propagation basis | Native sense |
|---|---|---|
| GCN / GIN (clique expansion) | Pairwise graph projection | Graph-native |
| AllSet / AllSetTransformer | Learned multiset maps $V \to E \to V$ | Incidence-native |
| Taha et al. (ICLR 2025) | Relational structures + influence graphs | Relation-native |
| **EO-Pattern MPNN (this work)** | Equal-edges walk + cardinality descriptors | Geometry-native |

---

## Project roadmap

- [x] Survey: higher-order structures and message passing frameworks
- [x] Survey: random walk variants and natural lifts to higher-order domains
- [x] Proposal: EO-pattern MPNN layer (mathematical construction)
- [ ] Implementation in PyTorch Geometric / TopoX
- [ ] Experiments: synthetic hypergraphs where clique expansion is provably lossy
- [ ] Comparison: GCN, GIN, AllSet, uniform hypergraph MPNN vs. EO-pattern layer
- [ ] Ablations: $P_{\mathrm{EE}}$ only / $\chi_{vu}$ only / full model

---

## Background surveys

Two background surveys accompany the main contribution:

**Survey 1 — Higher-order structures in graph learning**
Covers: hypergraphs, simplicial/cellular/combinatorial complexes, the MPNN formalism, WL expressivity, oversmoothing, oversquashing, CCNNs, CTNNs, and the intrinsic-vs-relational design tension.

**Survey 2 — Random walk variants and higher-order lifts**
Covers: Markov chain foundations, simple/lazy/weighted/restart/multi-step/non-backtracking walks on graphs, and their natural lifts to hypergraphs and simplicial, cellular, and combinatorial complexes. The guiding operator-centric perspective: *a choice of random walk is a choice of propagation geometry.*

---

## References

- Eidi, M. & Otter, N. (2025). *Geometric characterisation of structural and regular equivalences in undirected (hyper)graphs.* arXiv:2512.24961.
- Taha, D., Chapman, J., Eidi, M., Devriendt, K., & Montúfar, G. (2025). *Demystifying topological message-passing with relational structures.* ICLR 2025. arXiv:2506.06582.
- Chien, E., Pan, C., Peng, J., & Milenkovic, O. (2022). *You Are AllSet: A multiset learning framework for hypergraph neural networks.* ICLR 2022.
- Carletti, T., Battiston, F., Cencetti, G., & Fanelli, D. (2020). *Random walks on hypergraphs.* Physical Review E, 101(2).
- Gilmer, J. et al. (2017). *Neural message passing for quantum chemistry.* ICML 2017.
- Hajij, M. et al. (2023). *Topological deep learning: Going beyond graph data.* arXiv:2206.00606.

---

## Supervision

Research project supervised by **Prof. Guido Montúfar** (UCLA Mathematics / MPI MiS).
