"""Entry point for the single-file build (PyInstaller).

relperm.exe without arguments opens the editor window; with arguments it works like the
`relperm` command (e.g. `relperm.exe scal report.xlsx -o out`).
"""

import os
import sys

# A windowed build has no console: print() would fail on sys.stdout = None.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from relperm.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
