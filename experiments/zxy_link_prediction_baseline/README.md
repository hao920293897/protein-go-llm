# ZXY Link Prediction Baseline

This directory contains Xiaoyu Zhang's validated heterogeneous graph link prediction baseline for GO annotation.

Scope:
- PPI high-confidence filtering plus ID mapping and deduplication
- Heterogeneous KG construction with protein, gene, GO, and InterPro nodes
- Multi-relation GNN link prediction baseline for:
  - `protein -> GO`
  - `protein -> InterPro`
  - `InterPro -> GO`
  - `gene -> gene`
- Training on full GO label space with `prop_annotations + terms.pkl`
- Evaluation on both:
  - full label space
  - `terms_zero_10.pkl` subset

Key files:
- `model/data_loader.py`: data loading, GO term filtering, and PPI parsing
- `model/protein_kg.py`: KG construction logic
- `model/gnn_link_prediction.py`: multi-relation link prediction runner
- `model/pipeline.py`: pipeline entry used for the validated runs
- `model/config_gnn_smoke_example.json`: smoke-test config
- `model/config_gnn_full_mf_example.json`: full MF config

Notes:
- These files are intentionally isolated from the repository's main `model/` code so the existing pipeline is not overwritten.
- The example configs preserve the server paths used during validation. Update paths before running in a different environment.
