# Business Entity Resolution — Methodology

## Problem statement

Given each Source 1 business and candidate records from Sources 2 and 3, predict all records that refer to the same real-world business. Matching uses the supplied names, addresses, country labels, and training labels only. The challenge objective is precision-heavy macro F0.5 and includes singletons.

## Data

The challenge specification describes tab-separated source files with `entity_id`, `business_name`, `business_address`, `country`, plus training labels with `source1_entity_id` and comma-separated `matched_entity_ids`. The checked-in workspace did not include `dataset/train/` or `dataset/test/`, so data schema and row counts could not be inspected against actual files. The current files in `output/` are explicitly placeholders.

## Methodology

### Blocking strategy

Blocking is not implemented in this feature-engineering handoff. The current `candidate_pairs.tsv` must be produced by the blocking owner and should represent the final candidate set actually scored. This matters for both candidate recall and the challenge requirement that final matches be a subset of candidates. No candidate recall, reduction ratio, or blocking quality result is claimed.

### Feature engineering

`code/business_entity_resolution/src/features.py` creates one feature row per candidate pair. It joins candidate IDs to source records, supports the brief's comma-separated candidate list as well as one candidate ID per row, and optionally labels rows from ground truth membership.

Name features:

- Token Jaccard: shared tokens divided by the union.
- Levenshtein distance and max-length-normalized distance: edit effort in raw and comparable form.
- Jaro-Winkler: character similarity that tolerates short prefix variations.
- Token sort and token set ratios: fuzzy similarity robust to order changes and extra tokens.
- TF-IDF cosine: word unigram/bigram overlap weighted by corpus rarity.
- Absolute and relative length differences, common-token count, and common-token ratio: name size and shared evidence.

Address features:

- Full-address token Jaccard, Levenshtein (raw and normalized), Jaro-Winkler, token sort/set, and TF-IDF cosine capture overlap and spelling/order variation.
- Postal-code match extracts digit runs of 5–10 digits; companion flags mark a missing code in either or both records.
- Leading street-number agreement compares initial digits where present.
- State-like token overlap is best-effort token overlap after the first token, with a missingness flag; it uses no gazetteer.
- Missing-component and sparse-address flags identify empty records and addresses with fewer than three tokens.

Other features:

- Country equality compares normalized strings directly and supports arbitrary/unseen labels such as France without a fixed vocabulary.
- Name/address missingness flags indicate whether either side is empty.

Normalization lowercases, folds accents to ASCII, maps `&` to `and`, standardizes punctuation to spaces, and expands common suffix abbreviations (`inc`, `ltd`, `pvt`, `corp`, `co`) to full tokens. Person A's normalization implementation was not present, so this is a minimal local fallback; swap `normalize_text` for the canonical function when available. No embedding model or external data is used.

## Experimental setup

The module includes nine deterministic synthetic cases for close matches, non-matches, reordered tokens, missing fields, accent folding, and an unseen country label. The CLI runs these checks with `python3 src/main.py --synthetic-demo`. Real train/validation splits and F0.5 were not computed because the training files are absent.

## Results

No real-data feature matrix, baseline model, feature ranking, or score is available from this checkout. Feature importance remains to be measured by Person C after candidate pairs and training files are supplied. The synthetic demonstration reports feature values only; it is not a performance estimate.

## Reproducibility

From `code/business_entity_resolution/`, install `requirements.txt` and run the synthetic demo or the real-data command shown in its README. Real-data inputs require the official source TSVs, ground truth for labels, and candidate pairs. The feature output is tab-separated with one row per pair and an `is_match` column when labels are supplied.

## Limitations and outstanding work

Digit-run postal extraction can mistake other address numbers for postcodes. State-like token detection is approximate, and transliteration or severely sparse addresses can weaken similarity. Pairwise edit/string features are calculated per candidate in Python and may be CPU-bound on very large candidate sets. Blocking, model training, validation, threshold selection, feature importance, test predictions, output validation, and final packaging require the missing data and respective pipeline components; placeholder output files are not valid submissions.
