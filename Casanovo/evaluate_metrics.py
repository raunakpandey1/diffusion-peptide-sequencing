"""
evaluate_metrics.py  —  Casanovo v5 evaluation module
======================================================
Implements the standard evaluation protocol used in the Casanovo paper
(Yilmaz et al., ICML 2022) and NovoBoard benchmark framework:

  Protocol: Matched-set evaluation with I/L equivalence
  -------------------------------------------------------
  • Evaluate only on spectra that have a database-search annotation (MaxQuant).
    Casanovo always sequences every spectrum; spectra that MaxQuant rejected as
    low-quality/unidentifiable are excluded from the evaluation denominator.
    This matches the evaluation design in all published de novo benchmarks.

  • Isoleucine (I) and Leucine (L) are treated as equivalent.
    They share identical mass (113.084 Da) and are indistinguishable by tandem
    mass spectrometry.  All published de novo tools apply this normalization.

Metrics (4 × 2 levels):
  Amino-acid level  — character-wise LCS, summed over matched spectra
  Peptide level     — exact full-sequence match, over matched spectra

  AA  Precision = Σ LCS(pred_i, gt_i) / Σ len(pred_i)
  AA  Recall    = Σ LCS(pred_i, gt_i) / Σ len(gt_i)
  AA  Accuracy  = Σ LCS(pred_i, gt_i) / Σ max(len(pred_i), len(gt_i))
  AA  F1        = 2·P·R / (P+R)

  Pep Precision = TP / (matched spectra)
  Pep Recall    = TP / (total GT spectra)
  Pep Accuracy  = TP / (TP + FP_matched + FN)
  Pep F1        = 2·P·R / (P+R)

Published benchmark context (Casanovo, Yilmaz et al. 2022):
  AA  Recall  :  59 – 83 % (dataset-dependent)
  Pep Recall  :  20 – 40 % (dataset-dependent)
  → This module targets the upper range via I/L normalisation.

Usage:
    from evaluate_metrics import compute_all_metrics, load_ground_truth, load_casanovo_predictions
    preds  = load_casanovo_predictions("casanovo_EV1_predictions.mztab")
    gt     = load_ground_truth("Database search output_Ecoli_EV_1.xlsx")
    result = compute_all_metrics(preds, gt)
    print_metrics_table(result)
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. SEQUENCE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_sequence(seq: str, il_equivalent: bool = True) -> str:
    """
    Normalise a single-letter amino-acid sequence for comparison.

    Steps applied:
      1. Upper-case
      2. Strip modification annotations  e.g.  C[Carbamidomethyl] → C
      3. If il_equivalent=True: replace all I → L
         (Ile and Leu share identical mass and are indistinguishable by MS/MS)

    Parameters
    ----------
    seq           : raw amino-acid string
    il_equivalent : apply I→L substitution (True by default, matches all
                    published de novo benchmarks)
    """
    s = seq.upper().strip()
    s = re.sub(r'\[.*?\]', '', s)   # strip [modification] annotations
    s = re.sub(r'\(.*?\)', '', s)   # strip (modification) annotations
    if il_equivalent:
        s = s.replace('I', 'L')
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_ground_truth(xlsx_path: str, il_equivalent: bool = True) -> pd.DataFrame:
    """
    Load MaxQuant database-search results as the ground truth reference.

    Parameters
    ----------
    xlsx_path    : path to 'Database search output_Ecoli_EV_*.xlsx'
    il_equivalent: apply I→L normalisation to sequences (default True)

    Returns
    -------
    pd.DataFrame  columns: scan_number (int), sequence (str), score (float), pep (float)
    """
    df = pd.read_excel(xlsx_path, sheet_name="msms")
    gt = df[["Scan number", "Sequence", "Score", "PEP"]].copy()
    gt.columns = ["scan_number", "sequence", "score", "pep"]
    gt["sequence"] = gt["sequence"].apply(lambda s: normalize_sequence(str(s), il_equivalent))
    gt = gt.dropna(subset=["sequence"])
    gt = gt[gt["sequence"].str.len() > 0]
    gt = gt.drop_duplicates(subset="scan_number", keep="first")
    return gt.reset_index(drop=True)


def load_casanovo_predictions(mztab_path: str, il_equivalent: bool = True) -> pd.DataFrame:
    """
    Parse Casanovo v5 .mztab output into a tidy DataFrame.

    Casanovo v5 mztab format:
      sequence           → single-letter AA codes (clean, no modifications)
      PSM_ID             → sequential integer  (NOT the scan number)
      spectra_ref        → "ms_run[1]:controllerType=0 controllerNumber=1 scan=XXXX"
      search_engine_score[1] → log probability  (negative, e.g. -0.993)
      opt_ms_run[1]_aa_scores → comma-separated per-residue confidence (0–1)
      opt_ms_run[1]_proforma  → ProForma notation with modifications

    Parameters
    ----------
    mztab_path   : path to Casanovo .mztab file
    il_equivalent: apply I→L normalisation (default True)

    Returns
    -------
    pd.DataFrame  columns: scan_number (int), predicted_sequence (str),
                           confidence (float 0-1), aa_scores (list[float])
    """
    records = []
    with open(mztab_path, "r") as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("PSH"):
                header = line.split("\t")[1:]
            elif line.startswith("PSM") and header is not None:
                fields = line.split("\t")[1:]
                row    = dict(zip(header, fields))

                # ── scan number from spectra_ref ──────────────────────────
                spectra_ref = row.get("spectra_ref", "")
                m = re.search(r"scan=(\d+)", spectra_ref)
                if m is None:
                    continue
                scan_number = int(m.group(1))

                # ── per-residue AA confidence (comma-separated) ───────────
                aa_field = next((v for k, v in row.items() if "aa_scores" in k), "")
                aa_scores = []
                for token in aa_field.split(","):
                    try:
                        aa_scores.append(float(token.strip()))
                    except ValueError:
                        break

                # ── overall confidence = mean of per-AA scores ────────────
                confidence = float(np.mean(aa_scores)) if aa_scores else 0.0

                # ── clean + normalise sequence ────────────────────────────
                seq = normalize_sequence(row.get("sequence", ""), il_equivalent)

                if len(seq) > 0:
                    records.append({
                        "scan_number":        scan_number,
                        "predicted_sequence": seq,
                        "confidence":         confidence,
                        "aa_scores":          aa_scores,
                    })

    if not records:
        return pd.DataFrame(columns=["scan_number","predicted_sequence","confidence","aa_scores"])

    df = (pd.DataFrame(records)
            .drop_duplicates(subset="scan_number", keep="first")
            .reset_index(drop=True))
    return df


def load_casanovo_csv(csv_path: str, il_equivalent: bool = True) -> pd.DataFrame:
    """Fallback loader for Casanovo CSV output."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {"peptide":"predicted_sequence","seq":"predicted_sequence",
              "scan":"scan_number","scan number":"scan_number",
              "score":"confidence","peptide score":"confidence"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "predicted_sequence" not in df.columns and "sequence" in df.columns:
        df = df.rename(columns={"sequence": "predicted_sequence"})
    df["predicted_sequence"] = df["predicted_sequence"].apply(
        lambda s: normalize_sequence(str(s), il_equivalent))
    df = df.dropna(subset=["scan_number","predicted_sequence"])
    df = df.drop_duplicates(subset="scan_number", keep="first")
    if "aa_scores" not in df.columns:
        df["aa_scores"] = [[] for _ in range(len(df))]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEQUENCE ALIGNMENT — LCS
# ─────────────────────────────────────────────────────────────────────────────

def lcs_length(seq_a: str, seq_b: str) -> int:
    """
    Longest Common Subsequence length — standard AA-level metric for de novo
    sequencing (Casanovo paper, NovoBoard).

    Space-optimised O(|a|·|b|) DP.
    """
    m, n   = len(seq_a), len(seq_b)
    prev   = [0] * (n + 1)
    curr   = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            curr[j] = (prev[j-1] + 1 if seq_a[i-1] == seq_b[j-1]
                       else max(prev[j], curr[j-1]))
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATCHING: inner-join on scan_number
# ─────────────────────────────────────────────────────────────────────────────

def match_predictions_to_gt(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join on scan_number.

    Returns DataFrame with:
      scan_number, predicted_sequence, confidence,
      ground_truth_sequence, gt_score, gt_pep
    """
    return predictions.merge(
        ground_truth.rename(columns={
            "sequence": "ground_truth_sequence",
            "score":    "gt_score",
            "pep":      "gt_pep",
        }),
        on="scan_number",
        how="inner",
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. AMINO-ACID–LEVEL METRICS  (over matched spectra)
# ─────────────────────────────────────────────────────────────────────────────

def compute_aa_metrics(merged: pd.DataFrame) -> Dict:
    """
    AA-level Precision, Recall, Accuracy, F1 — computed over matched spectra.

    AA Precision = Σ LCS / Σ len(pred)
    AA Recall    = Σ LCS / Σ len(gt)
    AA Accuracy  = Σ LCS / Σ max(len(pred), len(gt))
    AA F1        = 2·P·R / (P+R)
    """
    total_lcs = total_pred = total_gt = total_max = 0
    per_spectrum = []

    for _, row in merged.iterrows():
        p, g = row["predicted_sequence"], row["ground_truth_sequence"]
        l    = lcs_length(p, g)
        total_lcs  += l
        total_pred += len(p)
        total_gt   += len(g)
        total_max  += max(len(p), len(g))
        per_spectrum.append({
            "scan_number":  row["scan_number"],
            "predicted":    p,
            "ground_truth": g,
            "lcs":          l,
            "pred_len":     len(p),
            "gt_len":       len(g),
            "aa_precision": l / len(p) if len(p) > 0 else 0.0,
            "aa_recall":    l / len(g) if len(g) > 0 else 0.0,
        })

    prec = total_lcs / total_pred if total_pred > 0 else 0.0
    rec  = total_lcs / total_gt   if total_gt   > 0 else 0.0
    acc  = total_lcs / total_max  if total_max  > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec)  if (prec+rec) > 0 else 0.0

    return {
        "aa_precision":    round(prec, 6),
        "aa_recall":       round(rec,  6),
        "aa_accuracy":     round(acc,  6),
        "aa_f1":           round(f1,   6),
        "_total_lcs":      total_lcs,
        "_total_pred_len": total_pred,
        "_total_gt_len":   total_gt,
        "_per_spectrum":   per_spectrum,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. PEPTIDE-LEVEL METRICS  (matched-set protocol)
# ─────────────────────────────────────────────────────────────────────────────

def compute_peptide_metrics(
    ground_truth: pd.DataFrame,
    merged: pd.DataFrame,
) -> Dict:
    """
    Peptide-level Precision, Recall, Accuracy, F1.

    Protocol (matches published benchmarks):
      TP  = exact sequence matches within the matched set
      FP  = wrong predictions in the matched set
            (spectra where Casanovo was sequenced but MaxQuant rejected
             are EXCLUDED from the denominator — they are not annotated,
             so they cannot be evaluated)
      FN  = GT spectra with no Casanovo prediction  +  wrong predictions

      Pep Precision = TP / len(merged)          [correct / all matched preds]
      Pep Recall    = TP / len(ground_truth)     [correct / all GT spectra]
      Pep Accuracy  = TP / (TP + FP + FN)
      Pep F1        = 2·P·R / (P+R)
    """
    total_gt      = len(ground_truth)
    matched       = len(merged)

    tp = int((merged["predicted_sequence"] == merged["ground_truth_sequence"]).sum())
    fp = matched - tp                           # wrong within matched set
    fn = (total_gt - matched) + fp              # unmatched GT + wrong

    prec = tp / matched    if matched    > 0 else 0.0
    rec  = tp / total_gt   if total_gt   > 0 else 0.0
    acc  = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

    return {
        "peptide_precision": round(prec, 6),
        "peptide_recall":    round(rec,  6),
        "peptide_accuracy":  round(acc,  6),
        "peptide_f1":        round(f1,   6),
        "_tp":               tp,
        "_fp":               fp,
        "_fn":               fn,
        "_matched_spectra":  matched,
        "_total_gt":         total_gt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. UNIFIED ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    predictions:          pd.DataFrame,
    ground_truth:         pd.DataFrame,
    confidence_threshold: float = 0.0,
) -> Dict:
    """
    Compute all 8 metrics (4 × 2 levels) for one sample.

    Evaluation protocol:
      • Matched-set: predictions evaluated only on GT-annotated scans
      • I/L equivalence already applied by load_* functions (default)
      • Optional confidence threshold to study precision–recall tradeoff

    Parameters
    ----------
    predictions          : output of load_casanovo_predictions()
    ground_truth         : output of load_ground_truth()
    confidence_threshold : keep only predictions with confidence ≥ this (0–1)
    """
    if confidence_threshold > 0:
        predictions = predictions[predictions["confidence"] >= confidence_threshold].copy()

    merged      = match_predictions_to_gt(predictions, ground_truth)
    aa_metrics  = compute_aa_metrics(merged)
    pep_metrics = compute_peptide_metrics(ground_truth, merged)

    return {
        "aa_precision":      aa_metrics["aa_precision"],
        "aa_recall":         aa_metrics["aa_recall"],
        "aa_accuracy":       aa_metrics["aa_accuracy"],
        "aa_f1":             aa_metrics["aa_f1"],
        "peptide_precision": pep_metrics["peptide_precision"],
        "peptide_recall":    pep_metrics["peptide_recall"],
        "peptide_accuracy":  pep_metrics["peptide_accuracy"],
        "peptide_f1":        pep_metrics["peptide_f1"],
        "_breakdown": {"aa": aa_metrics, "peptide": pep_metrics},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. REPORTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_table(results: Dict, sample_name: str = "") -> None:
    title = f"  Evaluation Results{f' — {sample_name}' if sample_name else ''}"
    print(f"\n{'='*62}\n{title}\n{'='*62}")
    print(f"{'Metric':<32} {'AA Level':>12} {'Peptide Level':>15}")
    print("─" * 62)
    for m in ["precision", "recall", "accuracy", "f1"]:
        aa  = results.get(f"aa_{m}",      float("nan"))
        pep = results.get(f"peptide_{m}", float("nan"))
        print(f"  {m.capitalize():<30} {aa:>12.4f} {pep:>15.4f}")
    print("=" * 62)
    bd  = results.get("_breakdown", {})
    pep = bd.get("peptide", {})
    if pep:
        print(f"\n  Peptide confusion (matched-set protocol):")
        print(f"    TP = {pep.get('_tp','?'):>6}  |  matched spectra = {pep.get('_matched_spectra','?')}")
        print(f"    FP = {pep.get('_fp','?'):>6}  |  total GT spectra = {pep.get('_total_gt','?')}")
        print(f"    FN = {pep.get('_fn','?'):>6}")
    print()


def save_results_csv(results: Dict, output_path: str, sample_name: str = "") -> None:
    row = {"sample": sample_name}
    for k in ["aa_precision","aa_recall","aa_accuracy","aa_f1",
              "peptide_precision","peptide_recall","peptide_accuracy","peptide_f1"]:
        row[k] = results.get(k, float("nan"))
    pd.DataFrame([row]).to_csv(output_path, index=False)
    print(f"  Saved metrics → {output_path}")


def save_per_spectrum_csv(results: Dict, output_path: str) -> None:
    per = results.get("_breakdown",{}).get("aa",{}).get("_per_spectrum",[])
    if per:
        pd.DataFrame(per).to_csv(output_path, index=False)
        print(f"  Saved per-spectrum → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Self-test …")
    gt = pd.DataFrame({
        "scan_number": [100,101,102,103,104],
        "sequence":    ["PEPTLDE","CASANOVA","ECOLI","PROTEIN","SEQTEST"],
        "score":       [120.,130.,90.,110.,100.],
        "pep":         [1e-5,1e-6,1e-4,1e-5,1e-3],
    })
    preds = pd.DataFrame({
        "scan_number":        [100,101,102,103,999],
        "predicted_sequence": ["PEPTIDE","CASANOVO","ECNLI","PROTIN","XXXXXX"],
        "confidence":         [0.95,0.88,0.72,0.81,0.55],
        "aa_scores":          [[],[],[],[],[]],
    })
    # Apply normalisation manually for the test (loaders do this automatically in production)
    gt["sequence"]            = gt["sequence"].apply(lambda s: normalize_sequence(s))
    preds["predicted_sequence"] = preds["predicted_sequence"].apply(lambda s: normalize_sequence(s))

    r = compute_all_metrics(preds, gt)
    print_metrics_table(r, "Synthetic")
    assert r["aa_precision"] > 0 and r["peptide_recall"] > 0
    print("All assertions passed ✓")
