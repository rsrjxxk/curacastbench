# CuraCastBench

Code and data for **CuraCastBench: Budgeted Link Prediction Evaluation in Versioned Knowledge Bases**.

The package implements the shared-budget release evaluator, recall decomposition (`endpoint_availability`, `pool_recall`, `inspection_reach`), bipartite graph baselines, the reaction-difference multi-token retriever, and the five full-catalog pair-scoring operators (`gpu`, `cpu`, `csr`, `sql`, `ann`).

## Files

- `src/curacast/`: shared-budget evaluator, graph rankers, reaction-difference model, and retrieval operators.
- `data/demo.json`: two versioned GitHub Advisory Database tasks (`2025-11 -> 2025-12` and `2025-12 -> 2026-01`).
- `config.json`: training hyperparameters and five-seed configuration.
- `run.py`: command-line entry point.

## Quick Start

Python 3.10+ (standard library only for `check`, `demo`, and `eval`):

```bash
python3 run.py check
python3 run.py demo
```

`check` verifies the rank-reversal construction (Proposition 1) and the multiplicative recall decomposition identity. `demo` evaluates `degree`, `coannotation`, and `adamic_adar` across the complete cutoff package populations (`10,444` and `10,525` queries) at budgets `N`, `3N`, and `10N` (`30, 37, 40` of `175` and `45, 41, 67` of `230` eligible additions at `10N`).

To score custom candidate lists on any versioned task:

```bash
python3 run.py eval --task task.json --lists lists.json --budgets 10000 50000
```

## Reaction-Difference Retrieval and Operators

Install numerical and deep-learning dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Train and run full-catalog inference on cutoff feature arrays (`sequence.npy`, `text.npy`, `reaction.npy`, `known.npy`, `valid.npy`):

```bash
python3 run.py train --data /path/to/features --out work/seed12.pt --seed 20261112
python3 run.py infer --data /path/to/features --checkpoint work/seed12.pt --out work/top100.npy
```

Benchmark physical pair-scoring operators (`gpu`, `cpu`, `csr`, `sql`, `ann`; `sql` uses `duckdb` and `pandas`, `ann` uses `faiss-cpu`):

```bash
python3 run.py bench --operator cpu --queries q.npy --targets z.npy --known known.npy --valid valid.npy
```

## Data Sources

Public upstream releases used in the benchmark:

- Rhea (`https://www.rhea-db.org/`) and UniProtKB/Swiss-Prot (`https://www.uniprot.org/`)
- Gene Ontology (`https://geneontology.org/`)
- Reactome (`https://reactome.org/`)
- DBLP Monthly XML Releases (`https://dblp.org/xml/release/`)
- GitHub Advisory Database (`https://github.com/github/advisory-database`, CC BY 4.0; see `NOTICE.md`)

This repository contains the core implementation and small runnable tasks. Large-scale release archives, precomputed feature matrices, and trained checkpoints are not included.
