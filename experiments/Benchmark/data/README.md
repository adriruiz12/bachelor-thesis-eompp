# Co-citation hypergraph datasets

These are the standard co-citation hypergraphs from HyperGCN
(Yadati et al., NeurIPS 2019), reused by AllSet (Chien et al., ICLR 2022).

## Provenance

Source: https://github.com/malllabiisc/HyperGCN
Commit: de049387d8d8c9fe6ebab83cb4b49cacfd4ed534
Path in source: data/cocitation/{cora,citeseer}/

Each dataset folder contains three pickle files:
  hypergraph.pickle : dict mapping hyperedge id -> list of node ids
  features.pickle   : node feature matrix (scipy sparse or dense)
  labels.pickle     : 1-D array of class labels

## File checksums (SHA-256)

cora/features.pickle    A20AB3C4903D6CF97D277F5CE8D26735E31D08C888CDD11E3F0A5BB78D7C0602
cora/hypergraph.pickle  49ABC7FFF72A07DE24E50D9D740D5CC63A32A40B2508995C16816FCB2729DE80
cora/labels.pickle      0E6EDD0D971DDE22C1116FA662362262163A7A6D557EAFD60AC6857DE4569527

citeseer/features.pickle    A1B1D07241C818E1551B1B4871BBF2116FE870773373372F9DBC5582C78236F4
citeseer/hypergraph.pickle  B8DEECBECEF20618D4E23AF5B81B1936B36810FC4565CA476353E8A6D7F5884D
citeseer/labels.pickle      C3368A596786DF4208F2777E1C8017B7A2A20307AEE8C7ED4F2571D9F529611D

## Note on splits/

`experiment_benchmark.py` builds its own class-stratified 50/25/25
train/validation/test splits via `data_benchmark.make_split` (one per seed).
These use the same proportions as AllSet, but stratify the nodes separately by class. The original
HyperGCN label-scarce splits (~20 labelled nodes per class) are not
included in this repository; a future run in that regime would need to be
built separately, since the current harness requires a validation set for
early stopping.