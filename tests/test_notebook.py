"""Keep the Colab driver survivable across a kernel restart.

A restart wipes the interpreter state the setup cells install -- the working
directory and the sys.path entry for src/ -- while the stored cell outputs stay
on screen, so the notebook still looks configured and the next in-kernel cell
dies with ModuleNotFoundError or a parquet path that resolves against /content.
The contract these tests hold: every cell that runs Python in the kernel first
re-runs notebooks/colab_bootstrap.py, and the install cell never asks for a
package version that would force a restart of its own.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "colab_pipeline.ipynb"
BOOTSTRAP = REPO_ROOT / "notebooks" / "colab_bootstrap.py"
STATUS = REPO_ROOT / "notebooks" / "colab_status.py"

# カーネル内セルが先頭で読み直してよいスクリプト。どちらも cwd と sys.path を戻す。
BOOTSTRAPPERS = ("colab_bootstrap.py", "colab_status.py")

# Drive のマウントとクローンのセルだけは対象外。リポジトリがまだ無い時点で動く
# セルなので、リポジトリ内の bootstrap を呼びようがない。
SETUP_MARKERS = ("drive.mount", "REPO_URL")


def code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def runs_in_kernel(source: str) -> bool:
    """True when the cell has Python that the kernel itself executes.

    `!cmd` forks a fresh process with its own cwd and `%magic` is handled by
    IPython, so a cell made only of those does not care about cwd or sys.path.
    Shell lines continued with a trailing backslash still belong to the shell.
    """
    continued = False
    for line in source.splitlines():
        stripped = line.strip()
        if continued:
            continued = stripped.endswith("\\")
            continue
        if not stripped or stripped.startswith(("#", "%")):
            continue
        if stripped.startswith("!"):
            continued = stripped.endswith("\\")
            continue
        return True
    return False


def test_in_kernel_cells_rebuild_the_kernel_state() -> None:
    checked = 0
    for source in code_cells():
        if not runs_in_kernel(source) or any(m in source for m in SETUP_MARKERS):
            continue
        checked += 1
        # colab_status.py は先頭で bootstrap を走らせる（下のテストで固定）。
        assert any(script in source for script in BOOTSTRAPPERS), (
            "in-kernel cell would break after a runtime restart:\n" + source
        )
    assert checked, "no in-kernel cells found -- the detector is wrong"


def test_the_status_script_bootstraps_before_it_reads_anything() -> None:
    source = STATUS.read_text(encoding="utf-8")
    assert "colab_bootstrap.py" in source
    assert source.index("colab_bootstrap.py") < source.index("def _lzh")


def test_the_install_cell_does_not_force_a_polars_upgrade() -> None:
    """Colab preinstalls polars 1.3x and imports it into the kernel early.

    Asking pip for a newer polars installs it on disk but cannot replace an
    already-imported module, so the notebook would need a runtime restart in
    the middle of a run. The code is kept working on 1.3x instead -- see the
    window-nesting note in model.normalise_by_race.
    """
    installs = [source for source in code_cells() if "pip -q install" in source]
    assert installs, "the dependency cell disappeared"
    for source in installs:
        assert "polars>=1.4" not in source.replace(" ", "")


def test_the_dependency_cell_installs_the_package_itself() -> None:
    """`!python -m kyotei....` forks; the kernel's sys.path does not follow it.

    Without an install those cells -- the download, the export, the backfill,
    every long-running one -- fail with ModuleNotFoundError. The install has to
    be editable: paths.PROJECT_ROOT walks up from the package file, so a copy
    under site-packages would point data/ away from the Drive symlink.
    """
    installs = [source for source in code_cells() if "pip -q install" in source]
    assert any("install -e" in source for source in installs)


def test_bootstrap_exports_src_on_pythonpath() -> None:
    """The fallback for a kernel where the editable install has not run yet."""
    program = (
        f"import runpy; runpy.run_path({str(BOOTSTRAP)!r});"
        "import os; print(os.environ['PYTHONPATH'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert str(REPO_ROOT / "src") in result.stdout.strip().split(os.pathsep)


def test_bootstrap_makes_kyotei_importable_from_any_cwd(tmp_path) -> None:
    program = (
        f"import runpy; runpy.run_path({str(BOOTSTRAP)!r});"
        "import pathlib, kyotei;"
        "print(pathlib.Path.cwd());"
        "print(kyotei.__file__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    cwd, module = result.stdout.strip().splitlines()[-2:]
    assert cwd == str(REPO_ROOT)
    assert module.startswith(str(REPO_ROOT / "src" / "kyotei"))


def test_status_runs_on_a_repo_with_nothing_fetched(tmp_path) -> None:
    """It has to survive the state it exists to diagnose: no data at all."""
    result = subprocess.run(
        [sys.executable, str(STATUS)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "== 現在地 ==" in result.stdout
    assert "== 次にやること ==" in result.stdout


def test_bootstrap_names_the_clone_cell_when_the_repo_is_missing(tmp_path) -> None:
    stray = tmp_path / "notebooks"
    stray.mkdir()
    copy = stray / BOOTSTRAP.name
    copy.write_text(BOOTSTRAP.read_text(encoding="utf-8"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(copy)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "2." in result.stderr and "clone" in result.stderr
