"""Build the editable WASM tutorial with every cell running on first load.

marimo 0.23.x exports ``runtime.auto_instantiate`` as false even when the
project setting is true.  This helper performs the normal WASM export, embeds
the initial outputs, and then enables that flag in the generated mount config.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "likelihood_population_tutorial.py"
OUTPUT_DIR = ROOT.parent / "docs" / ROOT.name / NOTEBOOK.stem
INDEX = OUTPUT_DIR / "index.html"

DISABLED = '"auto_instantiate": false'
ENABLED = '"auto_instantiate": true'


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html-wasm",
            str(NOTEBOOK),
            "-o",
            str(OUTPUT_DIR),
            "--mode",
            "edit",
            "--execute",
            "--force",
            "--no-sandbox",
        ],
        cwd=ROOT,
        check=True,
    )

    html = INDEX.read_text(encoding="utf-8")
    occurrences = html.count(DISABLED)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one disabled auto_instantiate setting in "
            f"{INDEX}, found {occurrences}. The marimo export format may "
            "have changed."
        )

    INDEX.write_text(html.replace(DISABLED, ENABLED), encoding="utf-8")
    print(f"Built {INDEX}")
    print("Enabled autorun on startup in the editable WASM export.")


if __name__ == "__main__":
    main()
