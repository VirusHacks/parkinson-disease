"""One-off helper: print one line per notebook cell."""
import json
import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

nb_path = Path(sys.argv[1] if len(sys.argv) > 1 else "colab_train_motorassist.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first = src.splitlines()[0] if src else ""
    print(f"[{i:02d}] {c['cell_type']:8s} | {first[:120]}")
