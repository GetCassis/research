"""White-box half of the experiment: read an open model at the token where a convention is decided.

The real model (Gemma 4 12B, loaded by `wb.hf.load_model`) needs a GPU; on a laptop only the
plumbing tests and the `info` step (tokenizer only) run. See the experiment README.
"""

import sys
from pathlib import Path

WB_DIR = Path(__file__).resolve().parents[1]  # src/whitebox
DEMO_DIR = WB_DIR.parent  # src: the behavioural package (talkdemo) and the scenarios

for _p in (str(DEMO_DIR), str(WB_DIR)):  # talkdemo + scenarios; wb
    if _p not in sys.path:
        sys.path.insert(0, _p)
