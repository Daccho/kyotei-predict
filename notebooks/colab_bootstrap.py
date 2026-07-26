"""Restore the kernel state that the notebook's in-process cells depend on.

A Colab kernel restarts under you -- idle timeout, OOM, crash, or a manual
restart -- and a restart throws away everything the setup cells put into the
interpreter: the working directory the clone cell chdir'd into, the sys.path
entry that makes `kyotei` importable, and the Drive mount that `data/` points
at. The stored cell outputs stay on screen, so the notebook still *looks* set
up.

Cells that shell out (`!cd ... && python -m kyotei....`) fork a fresh process
and are immune, which is why the damage only ever surfaces in the handful of
in-kernel cells -- as `ModuleNotFoundError: No module named 'kyotei'`, or as a
parquet file that suddenly does not exist.

Run this at the top of every in-kernel cell:

    import runpy
    runpy.run_path('/content/kyotei-predict/notebooks/colab_bootstrap.py')

It is idempotent and costs nothing to repeat.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys

# クローン先を決め打ちせず自分の位置から辿る。Colab では /content/kyotei-predict
# だが、ローカルで同じ手順を確かめるときも同じスクリプトが使える。
REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"

if not (SRC / "kyotei").is_dir():
    raise RuntimeError(
        f"{SRC} がありません。先に「2. リポジトリと依存関係」のセル (clone) を実行してください"
    )

os.chdir(REPO)

# sys.path.insert() は存在しないパスでも黙って通るので、実体を確かめた後に入れる。
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 存在しないパスを一度探索すると sys.path_importer_cache に None が残り、
# あとからクローンしても同じカーネルでは import が通らない。捨ててから import する。
importlib.invalidate_caches()

# data/ は Drive 上の実体へのシンボリックリンク。再起動でマウントが外れると
# リンクだけが残り、「parquet が無い」という原因から遠い症状で落ちる。
_data = REPO / "data"
if _data.is_symlink() and not _data.resolve().is_dir():
    raise RuntimeError(
        f"{_data} のリンク先 ({os.readlink(_data)}) がありません。"
        "Drive のマウントが外れています。「1. Drive をマウント」のセルから流し直してください"
    )

print("cwd:", REPO)
