"""alloc.lib.cycle_signals — Health signal tree viewer.

Reads dashboard JSON metadata (produced by :mod:`alloc.lib.dashboard`)
and renders a human-readable tree view of modules and their signals.

Usage
-----
    from alloc.lib.cycle_signals import render_tree, load_and_render
    tree = render_tree(metadata_dict)
    print(tree)

Or from CLI::

    python -m alloc.lib.cycle_signals [--json PATH]

Signals
-------
* **S1** — no tests
* **S2** — oversized (>200 lines, >15 functions)
* **S3** — dead code (0 imports)
* **S4** — lint/type errors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Signal labels
# ---------------------------------------------------------------------------

SIGNAL_LABELS: dict[str, str] = {
    "S1": "⚠  no tests",
    "S2": "⚠  oversized",
    "S3": "⚠  dead code",
    "S4": "⚠  lint errors",
}


def _signal_label(signal: str) -> str:
    """Return a human-readable label for a signal string like ``S1:no_tests``."""
    key = signal.split(":")[0]
    return SIGNAL_LABELS.get(key, f"⚠  {signal}")


# ---------------------------------------------------------------------------
# Tree rendering
# ---------------------------------------------------------------------------


def render_tree(metadata: dict[str, Any]) -> str:
    """Render a tree view from dashboard metadata dict.

    Parameters
    ----------
    metadata : dict
        The deserialised JSON output of :func:`dashboard.generate_json`.

    Returns
    -------
    str
        Multi-line tree view string.
    """
    lines: list[str] = []

    pkg = metadata.get("package", "unknown")
    total_modules = metadata.get("total_modules", 0)
    total_lines = metadata.get("total_lines", 0)
    total_functions = metadata.get("total_functions", 0)
    total_classes = metadata.get("total_classes", 0)
    total_tests = metadata.get("total_tests", 0)
    signals_summary = metadata.get("signals_summary", {})

    lines.append("╔══════════════════════════════════════════════════╗")
    lines.append("║  alloc Health Dashboard                          ║")
    lines.append("╠══════════════════════════════════════════════════╣")
    lines.append(f"║  Package:     {pkg:<36s}║")
    lines.append(f"║  Modules:     {total_modules:<36d}║")
    lines.append(f"║  Lines:       {total_lines:<36d}║")
    lines.append(f"║  Functions:   {total_functions:<36d}║")
    lines.append(f"║  Classes:     {total_classes:<36d}║")
    lines.append(f"║  Tests:       {total_tests:<36d}║")
    lines.append("╠══════════════════════════════════════════════════╣")

    if signals_summary:
        lines.append("║  Signals:                                       ║")
        for sig_key, count in sorted(signals_summary.items()):
            label = SIGNAL_LABELS.get(sig_key, sig_key)
            lines.append(
                f"║    {sig_key}: {count:>3}  {label:<26s}║"
            )
    else:
        lines.append("║  Signals:     none (all clear)                  ║")

    lines.append("╚══════════════════════════════════════════════════╝")
    lines.append("")

    # Module tree
    modules = metadata.get("modules", [])
    if not modules:
        lines.append("  (no modules found)")
        return "\n".join(lines)

    lines.append("┌─ Modules")

    for idx, mod in enumerate(modules):
        is_last = idx == len(modules) - 1
        prefix = "└── " if is_last else "├── "
        connector = "    " if is_last else "│   "

        mod_path = mod.get("path", "?")
        mod_funcs = mod.get("functions", 0)
        mod_classes = mod.get("classes", 0)
        mod_tests = mod.get("test_count", 0)
        mod_signals = mod.get("signals", [])
        has_docstring = mod.get("has_docstring", False)
        mod_lines = mod.get("lines", 0)

        # Build status badge
        badges: list[str] = []
        if not has_docstring:
            badges.append("no-doc")
        if mod_tests == 0:
            badges.append("no-tests")
        if mod_lines > 200:
            badges.append(f"{mod_lines}L")
        if mod_funcs > 15:
            badges.append(f"{mod_funcs}F")

        badge_str = ""
        if badges:
            badge_str = f" [{', '.join(badges)}]"

        lines.append(
            f"{prefix}{mod_path} ({mod_funcs}f, {mod_classes}c, "
            f"{mod_tests}t){badge_str}"
        )

        if mod_signals:
            for sig in mod_signals:
                label = _signal_label(sig)
                lines.append(f"{connector}  ⚡ {label}")

    lines.append("")
    return "\n".join(lines)


def load_and_render(json_path: str | Path) -> str:
    """Load dashboard JSON from *json_path* and render tree view.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file produced by :func:`dashboard.generate_json`.

    Returns
    -------
    str
        Rendered tree view.

    Raises
    ------
    FileNotFoundError
        If *json_path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Dashboard JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return render_tree(data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for cycle_signals.

    Usage::

        python -m alloc.lib.cycle_signals [--json PATH]

    If ``--json`` is provided, reads from that file.  Otherwise generates
    fresh metadata via :mod:`alloc.lib.dashboard`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="alloc health dashboard tree view"
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=str,
        default=None,
        help="Path to pre-generated dashboard JSON",
    )
    args = parser.parse_args()

    if args.json_path:
        tree = load_and_render(args.json_path)
    else:
        from alloc.lib.dashboard import generate_json

        json_str = generate_json()
        tree = render_tree(json.loads(json_str))

    print(tree)


if __name__ == "__main__":
    main()
