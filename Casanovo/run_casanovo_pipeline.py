"""
run_casanovo_pipeline.py  ─  Casanovo v5.x  (corrected CLI flags)
==================================================================
Key changes vs earlier versions:
  --output  →  --output_dir <dir>  +  --output_root <stem>

Model weights (casanovo_v5_0_0.ckpt, 548 MB) must be downloaded manually:
  curl -L -o casanovo_v5_0_0.ckpt \\
    https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt

Usage
-----
  python run_casanovo_pipeline.py                        # full run, both samples
  python run_casanovo_pipeline.py --sample EV1
  python run_casanovo_pipeline.py --skip_casanovo        # evaluate existing outputs
  python run_casanovo_pipeline.py --confidence_threshold 0.9


"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR  = SCRIPT_DIR / "casanovo_outputs"
CONFIG_PATH = SCRIPT_DIR / "casanovo_config.yaml"
MODEL_PATH  = SCRIPT_DIR / "casanovo_v5_0_0.ckpt"   # downloaded from v5.0.0 release

sys.path.insert(0, str(SCRIPT_DIR))
from evaluate_metrics import (
    load_ground_truth,
    load_casanovo_predictions,
    load_casanovo_csv,
    compute_all_metrics,
    print_metrics_table,
    save_results_csv,
    save_per_spectrum_csv,
)

# Each sample: mzML input, ground-truth xlsx, and the output stem casanovo will use
SAMPLES = {
    "EV1": {
        "mzml":         SCRIPT_DIR / "Ecoli_EV_1.mzML",
        "xlsx":         SCRIPT_DIR / "Database search output_Ecoli_EV_1.xlsx",
        "output_root":  "casanovo_EV1_predictions",          # stem passed to --output_root
        "metrics_out":  OUTPUT_DIR / "metrics_EV1.csv",
        "per_spec_out": OUTPUT_DIR / "per_spectrum_EV1.csv",
    },
    "EV2": {
        "mzml":         SCRIPT_DIR / "Ecoli_EV_2.mzML",
        "xlsx":         SCRIPT_DIR / "Database search output_Ecoli_EV_2.xlsx",
        "output_root":  "casanovo_EV2_predictions",
        "metrics_out":  OUTPUT_DIR / "metrics_EV2.csv",
        "per_spec_out": OUTPUT_DIR / "per_spectrum_EV2.csv",
    },
}


def mztab_path(sample_cfg: dict) -> Path:
    """Return the .mztab file casanovo will write for this sample."""
    return OUTPUT_DIR / f"{sample_cfg['output_root']}.mztab"


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHECK INSTALLATION
# ─────────────────────────────────────────────────────────────────────────────

def check_casanovo() -> bool:
    r = subprocess.run(["casanovo", "--version"], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"[✓] {(r.stdout or r.stderr).strip()}")
        return True
    print("[✗] casanovo not found. Install: pip install casanovo")
    return False


def check_model(model_path: Path) -> bool:
    if model_path.exists():
        size_mb = model_path.stat().st_size // 1024 // 1024
        print(f"[✓] Model weights: {model_path.name} ({size_mb} MB)")
        return True
    print(f"[✗] Model weights not found: {model_path}")
    print("    Download with:")
    print("    curl -L -o casanovo_v5_0_0.ckpt \\")
    print("      https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. RUN CASANOVO  (v5 CLI: --output_dir + --output_root)
# ─────────────────────────────────────────────────────────────────────────────

def run_casanovo(mzml_path: Path, output_root: str, model_path: Path) -> bool:
    """
    casanovo sequence
        --model      casanovo_v5_0_0.ckpt
        --output_dir casanovo_outputs/
        --output_root casanovo_EV1_predictions
        [--config    casanovo_config.yaml]
        Ecoli_EV_1.mzML

    Output file: casanovo_outputs/casanovo_EV1_predictions.mztab
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "casanovo", "sequence",
        "--model",       str(model_path),
        "--output_dir",  str(OUTPUT_DIR),
        "--output_root", output_root,
    ]
    if CONFIG_PATH.exists():
        cmd += ["--config", str(CONFIG_PATH)]
    cmd.append(str(mzml_path))

    print(f"\n[…] {mzml_path.name} → {output_root}.mztab")
    print("    " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        out = mztab_path({"output_root": output_root})
        print(f"[✓] Saved → {out}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[✗] Exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("[✗] casanovo not found. Run: pip install casanovo")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3–5. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_sample(name: str, cfg: dict, confidence_threshold: float = 0.0) -> dict:
    print(f"\n{'─'*60}\n  Evaluating: {name}\n{'─'*60}")

    gt = load_ground_truth(str(cfg["xlsx"]))
    print(f"  Ground truth : {len(gt)} spectra, {gt['sequence'].nunique()} unique seqs")

    pred_file = mztab_path(cfg)
    if not pred_file.exists():
        # Try CSV fallback
        csv_fb = pred_file.with_suffix(".csv")
        if csv_fb.exists():
            pred_file = csv_fb
        else:
            print(f"  [!] No predictions file found at {pred_file}")
            print(f"      Run Casanovo first.")
            return {}

    print(f"  Predictions  : {pred_file.name}")
    try:
        preds = (load_casanovo_predictions(str(pred_file))
                 if pred_file.suffix == ".mztab"
                 else load_casanovo_csv(str(pred_file)))
    except Exception as e:
        print(f"  [!] Parse error: {e}")
        return {}

    print(f"  Loaded {len(preds)} predictions")
    if confidence_threshold > 0:
        before = len(preds)
        preds = preds[preds["confidence"] >= confidence_threshold]
        print(f"  After filter ≥{confidence_threshold}: {len(preds)}/{before}")

    results = compute_all_metrics(preds, gt)
    print_metrics_table(results, sample_name=name)

    cfg["metrics_out"].parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(results, str(cfg["metrics_out"]), sample_name=name)
    save_per_spectrum_csv(results, str(cfg["per_spec_out"]))
    return results


def combined_summary(all_results: dict) -> None:
    if not all_results:
        return
    keys = [
        "aa_precision", "aa_recall", "aa_accuracy", "aa_f1",
        "peptide_precision", "peptide_recall", "peptide_accuracy", "peptide_f1",
    ]
    print(f"\n{'='*70}\n  COMBINED — E. coli EV Proteomics\n{'='*70}")
    print(f"{'Metric':<32}" + "".join(f"{s:>14}" for s in all_results))
    print("─" * (32 + 14 * len(all_results)))
    for k in keys:
        lvl, _, nm = k.partition("_")
        lbl = f"  [{lvl.upper():^3}] {nm.capitalize()}"
        row = f"{lbl:<32}" + "".join(
            f"{r.get(k, float('nan')):>14.4f}" for r in all_results.values()
        )
        print(row)
    print(f"{'='*70}\n")

    rows = [{"sample": s, **{k: r.get(k) for k in keys}}
            for s, r in all_results.items() if r]
    if rows:
        out = OUTPUT_DIR / "metrics_combined.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Casanovo v5 pipeline — E. coli EV")
    p.add_argument("--sample", choices=["EV1", "EV2", "both"], default="both")
    p.add_argument("--skip_casanovo", action="store_true",
                   help="Skip sequencing step, evaluate existing .mztab files")
    p.add_argument("--confidence_threshold", type=float, default=0.0)
    args = p.parse_args()

    print("\n" + "="*60)
    print("  Casanovo v5 Pipeline — E. coli Vesicle Proteomics")
    print("="*60 + "\n")

    if not args.skip_casanovo:
        if not check_casanovo():
            sys.exit(1)
        if not check_model(MODEL_PATH):
            sys.exit(1)

    samples = list(SAMPLES) if args.sample == "both" else [args.sample]

    if not args.skip_casanovo:
        for name in samples:
            cfg = SAMPLES[name]
            if cfg["mzml"].exists():
                run_casanovo(cfg["mzml"], cfg["output_root"], MODEL_PATH)
            else:
                print(f"[!] {cfg['mzml']} not found")

    all_results = {}
    for name in samples:
        cfg = SAMPLES[name]
        if cfg["xlsx"].exists():
            all_results[name] = evaluate_sample(name, cfg, args.confidence_threshold)
        else:
            print(f"[!] {cfg['xlsx']} not found")

    combined_summary(all_results)


if __name__ == "__main__":
    main()