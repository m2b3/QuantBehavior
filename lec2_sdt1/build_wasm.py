"""Build the editable WASM tutorial with every cell running on first load.

marimo 0.23.x intentionally exports only display/sharing preferences, so an
editable WASM export otherwise resets ``runtime.auto_instantiate`` to false.
This helper performs the normal export and then enables that runtime flag in
the generated mount configuration.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "sdt_gaussian_tutorial.py"
OUTPUT_DIR = ROOT.parent / "docs" / ROOT.name / "sdt_gaussian_tutorial"
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
    print("Enabled autorun on startup in the editable WASM export.")


if __name__ == "__main__":
    main()
