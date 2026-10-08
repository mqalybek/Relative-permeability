"""Entry point for the single-file build (PyInstaller).

relperm.exe without arguments opens the editor window; with arguments it works like the
`relperm` command (e.g. `relperm.exe scal report.xlsx -o out`).
`relperm.exe --selftest` opens the window, closes it after a few seconds and exits 0
(used by CI to prove the window really starts).

A windowed exe has no console, so any start-up error is written to relperm_error.log
next to the exe (or in %TEMP%) and shown in a message box instead of vanishing.
"""

import os
import sys
import tempfile
import traceback

# A windowed build has no console: print() would fail on sys.stdout = None.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _report(exc_text: str) -> str:
    """Write the traceback to a log file and show it in a message box; return the path."""
    base = (os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False)
            else os.getcwd())
    folders = [base, tempfile.gettempdir()]
    path = ""
    for folder in folders:
        try:
            path = os.path.join(folder, "relperm_error.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write(exc_text)
            break
        except OSError:
            path = ""
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("relperm: ошибка запуска",
                             exc_text[-1500:] + (f"\n\nПодробности: {path}" if path else ""))
        root.destroy()
    except Exception:
        pass
    return path


def _selftest():
    import matplotlib.pyplot as plt

    from relperm.app import RelPermApp

    app = RelPermApp()
    timer = app.fig.canvas.new_timer(interval=3000)
    timer.add_callback(plt.close, "all")
    timer.start()
    print("backend:", plt.get_backend())
    plt.show()
    print("selftest ok")


def run():
    gui = len(sys.argv) == 1 or sys.argv[1:] == ["--selftest"]
    if gui and "MPLBACKEND" not in os.environ:
        import matplotlib

        matplotlib.use("TkAgg")  # the window toolkit bundled with the exe
    if sys.argv[1:] == ["--selftest"]:
        _selftest()
        return
    from relperm.cli import main

    main()


if __name__ == "__main__":
    try:
        run()
    except SystemExit:
        raise
    except BaseException:
        _report(traceback.format_exc())
        sys.exit(1)
