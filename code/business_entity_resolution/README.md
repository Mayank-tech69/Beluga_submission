# Business entity resolution feature engineering

This package creates row-per-candidate pair features from the supplied TSV files. It uses only record text and the supplied training labels; it performs no external lookups and uses no embedding model.

## Data availability

The checked-in workspace currently does not contain `dataset/train/` or `dataset/test/`. The top-level `output/candidate_pairs.tsv` and `output/matching_results.tsv` are placeholders, not usable challenge files. Therefore an actual dataset feature matrix, validation score, feature-importance report, and test submission cannot be generated from this checkout. Add the official files to `dataset/` before using the real-data command.

## Run

From this directory, install pinned dependencies and run the synthetic demonstration:

```bash
python3 -m pip install -r requirements.txt
python3 src/main.py --synthetic-demo
```

Generate training features (candidate rows may use either one `candidate_entity_id` per row or a comma-separated `candidate_entity_ids` column):

```bash
python3 src/main.py \
  --pairs ../../output/candidate_pairs.tsv \
  --source1 ../../dataset/train/train_source1.tsv \
  --source2 ../../dataset/train/train_source2.tsv \
  --source3 ../../dataset/train/train_source3.tsv \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --output ../../output/train_features.tsv
```

The candidate-pairs file header must be `source1_entity_id` and `candidate_entity_ids` (or `candidate_entity_id`). For inference, omit `--ground-truth`. Source files must contain `entity_id`, `business_name`, `business_address`, and `country`; the ground truth must contain `source1_entity_id` and `matched_entity_ids`. All reads and writes use explicit tab separators.

## Feature method

Text is normalized locally to lowercase ASCII, ampersand to “and”, punctuation to whitespace, and common legal suffix abbreviations (`inc`, `ltd`, `pvt`, `corp`, `co`) to their full token forms. This is a fallback because Person A's canonical normalization code is absent; replace `normalize_text` with that canonical function when supplied. Similarities include token Jaccard, raw and normalized Levenshtein, Jaro-Winkler, token sort/set ratios, lengths, common-token measures, corpus-fitted word/unigram-bigram TF-IDF cosine, address postal-code and leading street-number agreement, state-like token overlap, missing/sparse-address indicators, and exact generic country-label equality. Country labels are never encoded against a fixed vocabulary.

The TF-IDF vectorizers fit over concatenated source records for each invocation and then compute sparse pairwise dot products. Edit/string distances are computed once per candidate pair in Python using RapidFuzz; very large candidate sets may therefore be CPU-bound. Memory use is linear in records plus sparse TF-IDF terms and output pair rows. Run in candidate batches if output memory becomes limiting.

## Limitations and handoff

State-like token overlap is deliberately gazetteer-free and is only a rough overlap signal; no postal format, country, or state list is hardcoded. Postal extraction accepts digit runs of 5–10 digits and may misread other numeric address components. Sparse addresses and transliterations reduce text similarity. Synthetic checks exercise high/low examples, missing values, France as an unseen label, and token-order changes. No real feature importance is claimed because the data and Person C model are unavailable. Blocking, the matching model, validation split, threshold tuning, complete `candidate_pairs.tsv`, `matching_results.tsv`, and final package validation remain upstream/downstream team work and require the missing challenge data and components.
