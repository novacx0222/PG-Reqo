"""Encode SQL CSVs to encode.pt only.

This is a memory-friendly wrapper around reqo_encode_sql.py. It keeps the
EXPLAIN/encoding behavior the same, but forces --no-save-original-reqo-dataset
so the heavier .npy conversion can be done later by reqo_pt_to_npy.py.
"""

from __future__ import annotations

import sys

from reqo_encode_sql import main


if __name__ == "__main__":
    if "--no-save-original-reqo-dataset" not in sys.argv:
        sys.argv.append("--no-save-original-reqo-dataset")
    main()
