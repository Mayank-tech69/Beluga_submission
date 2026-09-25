# Business Entity Resolution

This directory contains the end-to-end pipeline for duplicate business entity matching.

## Structure

- `src/`: source code for data loading, blocking, matching, and evaluation
- `requirements.txt`: pinned dependencies for reproducible execution

## Reproduction workflow

1. Prepare the raw data files.
2. Run preprocessing and blocking.
3. Generate candidate pairs.
4. Score and resolve matches.
5. Export final results into `output/matching_results.tsv`.

## Expected outputs

- `output/candidate_pairs.tsv`: all blocking candidates generated during the retrieval phase
- `output/matching_results.tsv`: final resolved matches exported for leaderboard submission

## Example commands

```bash
cd code/business_entity_resolution
python -m pip install -r requirements.txt
python src/main.py
```

Replace the example script entry point with your actual training/inference pipeline if needed.
