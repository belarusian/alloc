"""alloc.__main__ — Enable ``python -m alloc`` invocation.

Delegates to :func:`alloc.cli.main` so that running the package as a
module exercises the multi-trial training workflow CLI.

Usage
-----
    python -m alloc --tickers AAPL,META \\
        --positions-values '{"AAPL": 50000, "META": 50000}'
"""

from __future__ import annotations

import sys

from alloc.cli import main

if __name__ == "__main__":
    sys.exit(main())
