"""Pairwise business entity features; all calculations use supplied text only."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from sklearn.feature_extraction.text import TfidfVectorizer

PAIR_COLUMNS = ("source1_entity_id", "candidate_entity_id")


def normalize_text(value: Any) -> str:
    """Minimal local fallback. Replace with Person A's canonical text when available."""
    if value is None or pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value).strip()
    aliases = {"inc": "incorporated", "ltd": "limited", "pvt": "private",
               "corp": "corporation", "co": "company"}
    return " ".join(aliases.get(token, token) for token in value.split())


def _tokens(value: str) -> set[str]:
    return set(value.split())


def _similarities(left: str, right: str, prefix: str) -> dict[str, float]:
    a, b = normalize_text(left), normalize_text(right)
    ta, tb = _tokens(a), _tokens(b)
    union = ta | tb
    common = ta & tb
    max_len = max(len(a), len(b))
    return {
        f"{prefix}_token_jaccard": len(common) / len(union) if union else 0.0,
        f"{prefix}_levenshtein": float(distance.Levenshtein.distance(a, b)),
        f"{prefix}_levenshtein_norm": distance.Levenshtein.distance(a, b) / max_len if max_len else 0.0,
        f"{prefix}_jaro_winkler": distance.JaroWinkler.similarity(a, b),
        f"{prefix}_token_sort_ratio": fuzz.token_sort_ratio(a, b) / 100.0,
        f"{prefix}_token_set_ratio": fuzz.token_set_ratio(a, b) / 100.0,
        f"{prefix}_length_abs_diff": float(abs(len(a) - len(b))),
        f"{prefix}_length_ratio": min(len(a), len(b)) / max_len if max_len else 1.0,
        f"{prefix}_common_token_count": float(len(common)),
        f"{prefix}_common_token_ratio": len(common) / max(1, min(len(ta), len(tb))),
    }


def _codes(value: str) -> set[str]:
    # Six-digit Indian codes and 5/9 digit US ZIP codes; generic digit-run extraction.
    return {x for x in re.findall(r"(?<!\d)\d{5,10}(?!\d)", str(value or ""))}


def _street_number(value: str) -> str:
    norm = normalize_text(value)
    return re.sub(r"\D.*$", "", norm) if norm and norm[0].isdigit() else ""


def _state_like_tokens(value: str) -> set[str]:
    # Best effort only: alphabetic tokens after the first address token, excluding digits.
    tokens = normalize_text(value).split()
    return {t for t in tokens[1:] if len(t) >= 2 and not t.isdigit()}


def _address_extras(a1: str, a2: str) -> dict[str, float]:
    p1, p2 = _codes(a1), _codes(a2)
    s1, s2 = _state_like_tokens(a1), _state_like_tokens(a2)
    t1, t2 = _tokens(normalize_text(a1)), _tokens(normalize_text(a2))
    return {
        "address_pincode_match": float(bool(p1 & p2)),
        "address_pincode_missing_either": float(not p1 or not p2),
        "address_pincode_missing_both": float(not p1 and not p2),
        "address_street_number_match": float(bool(_street_number(a1)) and _street_number(a1) == _street_number(a2)),
        "address_state_like_overlap": len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0.0,
        "address_state_like_missing_either": float(not s1 or not s2),
        "address_components_missing_either": float(not t1 or not t2),
        "address_sparse_either": float(len(t1) < 3 or len(t2) < 3),
        "address_sparse_both": float(len(t1) < 3 and len(t2) < 3),
    }


def compute_features(name1: Any, addr1: Any, country1: Any,
                     name2: Any, addr2: Any, country2: Any) -> dict[str, float]:
    """Compute non-TF-IDF pair features. TF-IDF is added corpus-wise by build_feature_matrix."""
    n1, n2 = normalize_text(name1), normalize_text(name2)
    a1, a2 = normalize_text(addr1), normalize_text(addr2)
    out = _similarities(n1, n2, "name")
    out.update(_similarities(a1, a2, "address"))
    out.update(_address_extras(a1, a2))
    c1 = normalize_text(country1)
    c2 = normalize_text(country2)
    out["country_equal"] = float(c1 == c2)
    out["name_missing_either"] = float(not n1 or not n2)
    out["address_missing_either"] = float(not a1 or not a2)
    return out


def _vectorizer_similarity(corpus: list[str], left_idx: np.ndarray, right_idx: np.ndarray,
                           prefix: str) -> np.ndarray:
    # Fit once over the observed source records, then sparse dot products only for pairs.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    if not any(text.strip() for text in corpus):
        return np.zeros(len(left_idx), dtype=float)
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return np.zeros(len(left_idx), dtype=float)
    dots = np.asarray(matrix[left_idx].multiply(matrix[right_idx]).sum(axis=1)).ravel()
    return dots


def build_feature_matrix(pairs: pd.DataFrame, source1: pd.DataFrame,
                         source2: pd.DataFrame, source3: pd.DataFrame,
                         ground_truth: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join candidate IDs to records and return pair features, optionally with is_match."""
    p = pairs.copy()
    if "candidate_entity_ids" in p:
        p["candidate_entity_id"] = p["candidate_entity_ids"].fillna("").astype(str).str.split(",")
        p = p.explode("candidate_entity_id")
    if "candidate_entity_id" not in p:
        raise ValueError("pairs needs candidate_entity_id or candidate_entity_ids")
    p["candidate_entity_id"] = p["candidate_entity_id"].fillna("").astype(str).str.strip()
    p = p[p["candidate_entity_id"].ne("")].copy()
    sources = pd.concat([source1, source2, source3], ignore_index=True).drop_duplicates("entity_id")
    if not {"entity_id", "business_name", "business_address", "country"}.issubset(sources.columns):
        raise ValueError("source TSVs require entity_id, business_name, business_address, country")
    lookup = sources.set_index("entity_id")
    for side, ids in (("left", "source1_entity_id"), ("right", "candidate_entity_id")):
        p[f"{side}_name"] = p[ids].map(lookup["business_name"])
        p[f"{side}_address"] = p[ids].map(lookup["business_address"])
        p[f"{side}_country"] = p[ids].map(lookup["country"])
    if p[["left_name", "right_name"]].isna().any().any():
        raise ValueError("candidate pair contains an entity_id absent from source files")
    feature_rows = [compute_features(r.left_name, r.left_address, r.left_country,
                                     r.right_name, r.right_address, r.right_country)
                    for r in p.itertuples(index=False)]
    features = pd.DataFrame(feature_rows, index=p.index)
    names = [normalize_text(x) for x in pd.concat([sources.business_name, pd.Series([""])]).tolist()]
    addresses = [normalize_text(x) for x in pd.concat([sources.business_address, pd.Series([""])]).tolist()]
    # Build corpus indices via entity lookup; one TF-IDF fit per field.
    idx = {eid: i for i, eid in enumerate(sources.entity_id.tolist())}
    li = p.source1_entity_id.map(idx).to_numpy(dtype=int)
    ri = p.candidate_entity_id.map(idx).to_numpy(dtype=int)
    features["name_tfidf_cosine"] = _vectorizer_similarity(names, li, ri, "name")
    features["address_tfidf_cosine"] = _vectorizer_similarity(addresses, li, ri, "address")
    result = pd.concat([p[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True),
                        features.reset_index(drop=True)], axis=1)
    if ground_truth is not None:
        truth = dict(zip(ground_truth.source1_entity_id.astype(str),
                         ground_truth.matched_entity_ids.fillna("").astype(str)))
        result["is_match"] = [int(cid in set(truth.get(sid, "").split(",")))
                              for sid, cid in zip(result.source1_entity_id, result.candidate_entity_id)]
    return result


def synthetic_smoke_check() -> pd.DataFrame:
    """Small deterministic feature check without depending on challenge data."""
    cases = [
        ("Acme Ltd", "12 Main St, Boston 02110", "US", "ACME Limited", "12 Main Street, Boston 02110", "US", 1),
        ("Blue River Cafe", "4 Park Rd, Pune 411001", "India", "Blue River Café", "4 Park Road Pune 411001", "India", 1),
        ("Northstar Systems", "8 Lake Ave, Paris 75001", "France", "North Star System", "8 Lake Avenue Paris 75001", "France", 1),
        ("Acme Ltd", "12 Main St Boston", "US", "Zenith Motors", "900 Hill Road Austin", "US", 0),
        ("", "", "US", "", "", "US", 1),
        ("Mira Foods", "Shop 2 Market Road", "India", "Mira Food", "Market Rd Shop 2", "India", 1),
        ("Orchid Hotel", "17 River Lane", "France", "Orchid Bakery", "90 Rue Paris", "France", 0),
        ("Delta Corp", "Building 5 Sector 4", "India", "Delta Corporation", "Bldg 5 Sector 4", "India", 1),
        ("Sunrise Clinic", None, "US", "Sunrise Clinics", "", "US", 1),
    ]
    s1_rows, s2_rows, pair_rows, truth_rows = [], [], [], []
    for i, (n1, a1, c1, n2, a2, c2, expected) in enumerate(cases):
        f = compute_features(n1, a1, c1, n2, a2, c2)
        if expected:
            assert f["name_token_set_ratio"] >= 0.5 or not normalize_text(n1)
        else:
            assert f["name_token_set_ratio"] < 0.9
        left_id, right_id = f"S1-{i:05d}", f"S2-{i:05d}"
        s1_rows.append({"entity_id": left_id, "business_name": n1, "business_address": a1,
                        "country": c1})
        s2_rows.append({"entity_id": right_id, "business_name": n2, "business_address": a2,
                        "country": c2})
        pair_rows.append({"source1_entity_id": left_id, "candidate_entity_ids": right_id})
        truth_rows.append({"source1_entity_id": left_id,
                           "matched_entity_ids": right_id if expected else ""})
    schema = ["entity_id", "business_name", "business_address", "country"]
    return build_feature_matrix(pd.DataFrame(pair_rows), pd.DataFrame(s1_rows), pd.DataFrame(s2_rows),
                                pd.DataFrame(columns=schema), pd.DataFrame(truth_rows))
