"""Report where the pipeline stands, and name the next cell to run.

Colab hides the state that matters. The notebook on screen is a different file
from the clone `git pull` updates, a restarted kernel keeps stale output on
display, and `data/` is a symlink into Drive that can come unmounted underneath
a running session. So when a cell dies on a missing parquet, the traceback
cannot tell you which of the three it is: wrong working directory, archives
never fetched, or fetched but never exported.

This walks the filesystem and says which one it is, so the answer costs one
cell instead of a round of guessing. Reading it also works from a notebook that
predates the fix, because the logic lives in the repo rather than in a cell:

    import runpy
    runpy.run_path('/content/kyotei-predict/notebooks/colab_status.py')
"""

from __future__ import annotations

import pathlib
import runpy
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

# 貼り付けただけでも動くように、cwd と sys.path はこちらで先に直す。
runpy.run_path(str(HERE / "colab_bootstrap.py"))

REPO = HERE.parent
DATA = REPO / "data"
PARQUET = DATA / "parquet"
ODDS = DATA / "odds" / "odds3t_2024.jsonl"


def _lzh(kind: str) -> int:
    """Count fetched archives of one kind. Drive is slow, so count once."""
    root = DATA / "raw" / kind
    return sum(1 for _ in root.rglob("*.lzh")) if root.is_dir() else 0


def _describe(path: pathlib.Path) -> str:
    if not path.exists():
        return "無し"
    return f"{path.stat().st_size / 1e6:.1f} MB"


def _shell_can_import() -> bool:
    """Can a `!python -m kyotei....` cell find the package?

    Those cells fork a fresh interpreter, which inherits the kernel's
    environment but none of its sys.path. So an `import kyotei` that works in
    the notebook says nothing about whether cell 4 will run -- only this does.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import kyotei"], capture_output=True
    )
    return probe.returncode == 0


daily = _lzh("B") + _lzh("K")
fan = _lzh("fan")
entries = PARQUET / "entries.parquet"
features = PARQUET / "features.parquet"

print("== 現在地 ==")
print(f"{'cwd':16}:", pathlib.Path.cwd())
print(f"{'data':16}:", DATA.resolve())
print(f"{'raw (B+K)':16}:", daily, "ファイル")
print(f"{'raw (fan)':16}:", fan, "ファイル")
for name in ("entries", "payouts", "races", "features"):
    print(f"{name + '.parquet':16}:", _describe(PARQUET / f"{name}.parquet"))
if ODDS.exists():
    with ODDS.open(encoding="utf-8") as handle:
        print(f"{'odds jsonl':16}:", sum(1 for _ in handle), "行")
else:
    print(f"{'odds jsonl':16}: 無し")

shell_ok = _shell_can_import()
print(f"{'shell (!python)':16}:", "kyotei が見える" if shell_ok else "見えない")

print()
print("== 次にやること ==")
if not shell_ok:
    print("2. リポジトリと依存関係のセル（editable install を含む）")
    print("   これが通るまで !python -m kyotei.... のセルは全部落ちる。")
elif daily == 0:
    print("4. データ取得（kyotei.download）→ 5. parquet 化（kyotei.export）")
    print("   全期間で約2.4時間。オッズのバックフィルだけでは entries は作られない。")
elif not entries.exists():
    print("5. parquet 化（kyotei.export）。LZH は取得済みなので数分で終わる。")
elif not features.exists():
    print("6. 特徴量のセル")
else:
    print("7. 学習（kyotei.model）")
