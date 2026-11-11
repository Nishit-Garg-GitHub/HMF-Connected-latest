"""Run the HMF–FDCA research workflow sequentially from one command.

Every stage consumes artifacts from the previous stage. The runner stops on
an error, writes a log for each script, and supports resuming from a selected
stage. It intentionally does not run the legacy standalone scripts in
parallel; they are retained under archive_legacy for provenance/reference.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
STAGES = [
    ("01_preprocess_data.py", "01 — cleaning and feature engineering"),
    ("02_kmeans.py", "02 — K-means clustering, K=3,4,5,6"),
    ("03_oversampling.py", "03 — ROS, SMOTE and ADASYN fold audit"),
    ("04_model_evaluation.py", "04 — DT, SVM, RF and NN comparison"),
    ("05_learning_curves.py", "05 — learning curves and generalisation"),
    ("06_shap_analysis.py", "06 — SHAP / feature importance"),
    ("07_regression.py", "07 — joint four-output regression"),
    ("08_nsga2.py", "08 — validation-gated NSGA-II"),
    ("09_final_report.py", "09 — integrated final report"),
]


def parse_args():
    """Parse optional resume and clean-run controls."""
    parser = argparse.ArgumentParser(description="Run the connected HMF/FDCA ML workflow.")
    parser.add_argument("--start", default="01", choices=[f"{i:02d}" for i in range(1, 10)],
                        help="Resume from this stage number (default: 01).")
    parser.add_argument("--no-clean", action="store_true",
                        help="Keep previous results; otherwise stage logs/status are reset.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable; defaults to the active environment.")
    return parser.parse_args()


def validate_inputs():
    """Fail early if the source workbook or expected scripts are missing."""
    source = SCRIPTS / "Nishith_Summary Table of different cell types.xlsx"
    if not source.exists():
        raise FileNotFoundError(f"Missing literature workbook: {source}")
    missing = [name for name, _ in STAGES if not (SCRIPTS / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing pipeline scripts: {missing}")


def run_stage(python: str, filename: str, description: str, index: int):
    """Execute one stage and persist stdout/stderr for troubleshooting."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / f"stage_{index:02d}_{Path(filename).stem}.log"
    command = [python, str(SCRIPTS / filename)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::FutureWarning"
    env["MPLBACKEND"] = "Agg"
    mpl_dir = Path(tempfile.gettempdir()) / "hmf_mpl_config"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    started = time.time()
    print("\n" + "=" * 86)
    print(f"STARTING {description}")
    print("Command:", " ".join(command))
    print("Log:", log_path)
    print("=" * 86, flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    elapsed = time.time() - started
    if code != 0:
        raise RuntimeError(
            f"Stage {index:02d} failed with exit code {code}. Read {log_path}."
        )
    print(f"COMPLETED {description} in {elapsed:.1f} seconds.", flush=True)
    return {"stage": index, "script": filename, "description": description,
            "seconds": round(elapsed, 2), "status": "completed", "log": str(log_path)}


def main():
    """Run stages in dependency order, never concurrently."""
    args = parse_args()
    validate_inputs()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.no_clean and args.start == "01":
        for pattern in ("stage_*.log", "kmeans_labels_K*.csv", "classification_*.csv",
                        "oversampling_audit.*", "learning_curve_selected_model.csv",
                        "feature_importance_selected_model.csv", "multioutput_*.csv",
                        "per_target_*.csv", "regressor_*.joblib",
                        "regression_validation_gate.json", "nsga2_*.csv", "nsga2_status.json",
                        "pipeline_manifest.json", "FINAL_RESEARCH_REPORT.md"):
            for path in RESULTS.glob(pattern):
                path.unlink(missing_ok=True)
    start_number = int(args.start)
    outcomes = []
    for index, (filename, description) in enumerate(STAGES, start=1):
        if index < start_number:
            continue
        outcomes.append(run_stage(args.python, filename, description, index))
    print("\nPIPELINE COMPLETED")
    print(f"Stages executed: {len(outcomes)}")
    print(f"Final report: {RESULTS / 'FINAL_RESEARCH_REPORT.md'}")
    print("Important: F1 is measured on unseen papers; no score is guaranteed.")


if __name__ == "__main__":
    main()
