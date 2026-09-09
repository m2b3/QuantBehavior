"""Run and compare the Python/Marimo, R, and MATLAB SDT lessons.

Usage:
    python check_sdt_consistency.py
    python check_sdt_consistency.py --require-all

The checker exports the Marimo notebook to a temporary flat Python script, runs
all available language versions with invisible/file-backed graphics, and compares
deterministic results. Random simulation counts are deliberately excluded because
the languages use different random-number generators even with the same seed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.stats import norm, skewnorm


ROOT = Path(__file__).resolve().parent

PATTERNS = {
    "roc_auc": re.compile(
        r"(?:Area under the theoretical ROC curve|Area Under the Curve \(AUC\)):\s*"
        r"([-+0-9.eE]+)"
    ),
    "observed": re.compile(
        r"Corrected d':\s*([-+0-9.eE]+);\s*corrected c:\s*([-+0-9.eE]+)",
        re.IGNORECASE,
    ),
    "two_afc": re.compile(
        r"(?:Probability correct in 2I-2AFC|"
        r"Prob\. of being correct in 2-I, 2-AFC task):\s*([-+0-9.eE]+)"
    ),
    "skew_auc": re.compile(
        r"(?:Skew-normal ROC AUC|Area Under the Curve \(AUC\)):\s*([-+0-9.eE]+)"
    ),
}

TOLERANCES = {
    "roc_auc": 6e-4,
    "observed_dprime": 1.1e-3,
    "observed_c": 1.1e-3,
    "two_afc": 6e-4,
    "skew_auc": 6e-4,
}


def run_command(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        details = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n{details}"
        )
    return completed.stdout + completed.stderr


def python_output() -> str:
    # Exporting executes the notebook's actual cells in dependency order without
    # starting a Marimo server. The exported file lives only in a temp directory.
    with tempfile.TemporaryDirectory(prefix="sdt_check_") as temp_directory:
        flat_script = Path(temp_directory) / "sdt_flat.py"
        run_command(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "script",
                "sdt.py",
                "-o",
                str(flat_script),
                "--force",
            ]
        )
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        return run_command([sys.executable, str(flat_script)], env=environment)


def r_output(rscript: str) -> str:
    expression = (
        "pdf(tempfile(fileext='.pdf')); "
        "source('sdt.R', echo=FALSE); "
        "invisible(dev.off())"
    )
    return run_command([rscript, "-e", expression])


def matlab_output(matlab: str, *, force_fallback: bool = False) -> str:
    expression = (
        "set(0,'DefaultFigureVisible','off'); "
        "run('sdt.m'); close all;"
    )
    environment = os.environ.copy()
    if force_fallback:
        environment["SDT_FORCE_FALLBACK"] = "1"
    else:
        environment.pop("SDT_FORCE_FALLBACK", None)
    return run_command([matlab, "-batch", expression], env=environment)


def parse_metrics(output: str, language: str) -> dict[str, float]:
    roc_matches = PATTERNS["roc_auc"].findall(output)
    observed_match = PATTERNS["observed"].search(output)
    two_afc_match = PATTERNS["two_afc"].search(output)
    skew_matches = PATTERNS["skew_auc"].findall(output)
    if not roc_matches or observed_match is None or two_afc_match is None or not skew_matches:
        raise RuntimeError(
            f"Could not parse all validation metrics from {language} output:\n{output}"
        )

    # Python uses the generic AUC label twice. The first value is the Gaussian
    # ROC and the last is the skew-normal ROC. R and MATLAB use distinct labels.
    return {
        "roc_auc": float(roc_matches[0]),
        "observed_dprime": float(observed_match.group(1)),
        "observed_c": float(observed_match.group(2)),
        "two_afc": float(two_afc_match.group(1)),
        "skew_auc": float(skew_matches[-1]),
    }


def reference_metrics() -> dict[str, float]:
    loc_noise, sd_noise = 40.0, 10.0
    loc_signal, sd_signal = 60.0, 10.0
    dprime = (loc_signal - loc_noise) / sd_noise

    hits, misses = 40, 10
    false_alarms, correct_rejections = 15, 135
    corrected_hit = (hits + 0.5) / (hits + misses + 1)
    corrected_fa = (false_alarms + 0.5) / (
        false_alarms + correct_rejections + 1
    )
    z_hit = norm.ppf(corrected_hit)
    z_fa = norm.ppf(corrected_fa)

    skew_thresholds = np.linspace(0, 100, 100)
    skew_tpr = 1 - skewnorm.cdf(skew_thresholds, -2, loc=65, scale=10)
    skew_fpr = 1 - skewnorm.cdf(skew_thresholds, 2, loc=35, scale=10)
    roc_x = np.r_[1.0, skew_fpr, 0.0]
    roc_y = np.r_[1.0, skew_tpr, 0.0]
    ordering = np.argsort(roc_x)
    skew_auc = np.trapezoid(roc_y[ordering], roc_x[ordering])

    return {
        "roc_auc": float(norm.cdf(dprime / math.sqrt(2))),
        "observed_dprime": float(z_hit - z_fa),
        "observed_c": float(-0.5 * (z_hit + z_fa)),
        "two_afc": float(norm.cdf(dprime / math.sqrt(2))),
        "skew_auc": float(skew_auc),
    }


def compare(results: dict[str, dict[str, float]]) -> None:
    reference = reference_metrics()
    failures: list[str] = []

    print("\nDeterministic cross-language results")
    languages = list(results)
    language_width = max(12, max(map(len, languages)) + 2)
    print(
        f"{'metric':<20}{'reference':>12}"
        + "".join(f"{name:>{language_width}}" for name in languages)
    )
    for metric, expected in reference.items():
        row = f"{metric:<20}{expected:>12.6f}"
        for language in languages:
            actual = results[language][metric]
            row += f"{actual:>{language_width}.6f}"
            if not math.isclose(actual, expected, abs_tol=TOLERANCES[metric]):
                failures.append(
                    f"{language} {metric}: {actual} versus reference {expected}"
                )
        print(row)

    for left_index, left in enumerate(languages):
        for right in languages[left_index + 1 :]:
            for metric in reference:
                if not math.isclose(
                    results[left][metric],
                    results[right][metric],
                    abs_tol=TOLERANCES[metric],
                ):
                    failures.append(
                        f"{left}/{right} disagree on {metric}: "
                        f"{results[left][metric]} versus {results[right][metric]}"
                    )

    if failures:
        raise AssertionError("Cross-language checks failed:\n- " + "\n- ".join(failures))
    print("\nAll available implementations agree within numerical tolerance.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail instead of skipping when Rscript or MATLAB is unavailable",
    )
    arguments = parser.parse_args()

    # Catch Python syntax errors before invoking Marimo's exporter.
    compile((ROOT / "sdt.py").read_text(encoding="utf-8"), "sdt.py", "exec")

    runners: list[tuple[str, object]] = [("Python", python_output)]
    optional = {
        "R": (shutil.which("Rscript"), r_output),
        "MATLAB": (shutil.which("matlab"), matlab_output),
    }
    for language, (executable, runner) in optional.items():
        if executable is None:
            message = f"{language} executable not found"
            if arguments.require_all:
                raise RuntimeError(message)
            print(f"Skipping {language}: {message}.")
            continue
        runners.append((language, lambda exe=executable, run=runner: run(exe)))

    results: dict[str, dict[str, float]] = {}
    for language, runner in runners:
        print(f"Running {language} checks...")
        output = runner()
        if "All internal consistency checks passed." not in output:
            raise RuntimeError(f"{language} did not report that its internal checks passed.")
        results[language] = parse_metrics(output, language)
        if language == "MATLAB" and "Palamedes fit" in output:
            print("Running MATLAB fallback checks...")
            fallback_output = matlab_output(shutil.which("matlab"), force_fallback=True)
            if "All internal consistency checks passed." not in fallback_output:
                raise RuntimeError("MATLAB fallback internal checks did not pass.")
            results["MATLAB-fallback"] = parse_metrics(
                fallback_output, "MATLAB fallback"
            )

    compare(results)


if __name__ == "__main__":
    main()
