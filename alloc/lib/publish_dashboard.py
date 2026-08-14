"""alloc.lib.publish_dashboard — HTML publisher for dashboard metadata.

Takes the JSON metadata produced by :mod:`alloc.lib.dashboard` and renders
a standalone, responsive HTML page with inline CSS/JS.  Signals S1–S4 are
colour-coded by severity.  An optional ``--sync`` CLI flag pushes the
generated HTML to a ``gh-pages`` branch for GitHub Pages hosting.

Supports single-package view and multi-package comparison mode.

Usage
-----
    from alloc.lib.publish_dashboard import generate_html, publish

    html = generate_html(metadata_dict)
    publish(html, output_path="dashboard.html", sync=False)

Or from CLI::

    python -m alloc.lib.publish_dashboard [--json PATH] [--output PATH] [--sync]
    python -m alloc.lib.publish_dashboard --compare alloc/docs_dashboard_metadata.json new-trader.json

Signals
-------
* **S1** — no tests (orange)
* **S2** — oversized (amber)
* **S3** — dead code (slate)
* **S4** — lint/type errors (red)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal colour palette
# ---------------------------------------------------------------------------

SIGNAL_COLORS: dict[str, tuple[str, str]] = {
    # (background, text)
    "S1": ("#fff3cd", "#856404"),   # orange — no tests
    "S2": ("#ffecb5", "#6d5a00"),   # amber — oversized
    "S3": ("#e2e3e5", "#383d41"),   # slate — dead code
    "S4": ("#f8d7da", "#721c24"),   # red — lint errors
}

SIGNAL_ICONS: dict[str, str] = {
    "S1": "🧪",
    "S2": "📦",
    "S3": "💀",
    "S4": "🔧",
}

SIGNAL_LABELS: dict[str, str] = {
    "S1": "No Tests",
    "S2": "Oversized",
    "S3": "Dead Code",
    "S4": "Lint Errors",
}


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def _escape_html(text: str) -> str:
    """Minimal HTML escaping for safe embedding."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _signal_badge(signal: str) -> str:
    """Return an HTML span for a single signal badge."""
    key = signal.split(":")[0]
    label = signal.split(":", 1)[1] if ":" in signal else key
    icon = SIGNAL_ICONS.get(key, "⚠️")
    return (
        f'<span class="badge badge-{key.lower()}">'
        f"{icon} {_escape_html(label)}</span>"
    )


def _module_card(mod: dict[str, Any]) -> str:
    """Return an HTML module card div."""
    path = _escape_html(mod.get("path", "?"))
    lines = mod.get("lines", 0)
    functions = mod.get("functions", 0)
    classes = mod.get("classes", 0)
    tests = mod.get("test_count", 0)
    signals = mod.get("signals", [])

    badges_html = " ".join(_signal_badge(s) for s in signals)
    if badges_html:
        status_html = badges_html
    else:
        status_html = '<span class="clear">clear</span>'

    border_color = "#22c55e" if not signals else "#c0392b"

    return (
        f'<div class="mod-card" style="border-color: {border_color};">'
        f'<div class="mod-header">'
        f'<span class="mod-name">{path}</span>'
        f'<span class="mod-signals">{status_html}</span>'
        f'</div>'
        f'<div class="mod-meta">'
        f'<span>CLASSES: {classes}</span>'
        f'<span>FUNCS: {functions}</span>'
        f'<span>LINES: {lines}</span>'
        f'<span>TESTS: {tests}</span>'
        f'</div>'
        f'</div>'
    )


def _summary_card(title: str, value: str, card_class: str = "") -> str:
    """Return an HTML stat card div."""
    cls = f"stat-card {card_class}" if card_class else "stat-card"
    return (
        f'<div class="{cls}">'
        f'<div class="stat-value">{_escape_html(value)}</div>'
        f'<div class="stat-label">{_escape_html(title)}</div>'
        f'</div>'
    )


def _signal_summary_cards(signals_summary: dict[str, int]) -> str:
    """Return HTML stat cards for each signal type count."""
    if not signals_summary:
        return '<div class="stat-card"><div class="stat-value">0</div>' \
               '<div class="stat-label">All Clear</div></div>'

    cards = []
    for key in sorted(signals_summary.keys()):
        count = signals_summary[key]
        label = SIGNAL_LABELS.get(key, key)
        cards.append(
            f'<div class="stat-card warn">'
            f'<div class="stat-value">{count}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def _comparison_section(packages: list[dict[str, Any]]) -> str:
    """Generate a comparison table between multiple packages."""
    rows = []
    metrics = [
        ("Modules", "total_modules"),
        ("Lines", "total_lines"),
        ("Functions", "total_functions"),
        ("Classes", "total_classes"),
        ("Tests", "total_tests"),
        ("S1: No Tests", "signals_summary.S1"),
        ("S2: Oversized", "signals_summary.S2"),
        ("S3: Dead Code", "signals_summary.S3"),
        ("S4: Lint Errors", "signals_summary.S4"),
    ]

    headers = "".join(
        f'<th>{_escape_html(pkg.get("package", "?"))}</th>'
        for pkg in packages
    )

    for label, key in metrics:
        cells = []
        for pkg in packages:
            if "." in key:
                parent, child = key.split(".")
                val = pkg.get(parent, {}).get(child, 0)
            else:
                val = pkg.get(key, 0)
            formatted = f"{val:,}" if isinstance(val, int) and val > 100 else str(val)
            cells.append(f'<td class="comp-val">{formatted}</td>')

        cells.insert(0, f'<td class="comp-label">{label}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        f'<div class="section">'
        f'<div class="section-title">Shape Comparison</div>'
        f'<div class="comp-table-wrapper">'
        f'<table class="comp-table">'
        f'<thead><tr><th class="comp-label">Metric</th>{headers}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        f'</table>'
        f'</div>'
        f'</div>'
    )


def generate_html(
    metadata: dict[str, Any],
    compare_with: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a standalone HTML dashboard page from metadata.

    Parameters
    ----------
    metadata : dict
        The deserialised JSON output of :func:`dashboard.generate_json`.
    compare_with : list of dict, optional
        Additional package metadata dicts to compare against.
    """
    pkg = _escape_html(metadata.get("package", "unknown"))
    total_modules = metadata.get("total_modules", 0)
    total_lines = metadata.get("total_lines", 0)
    total_functions = metadata.get("total_functions", 0)
    total_classes = metadata.get("total_classes", 0)
    total_tests = metadata.get("total_tests", 0)
    signals_summary = metadata.get("signals_summary", {})
    modules = metadata.get("modules", [])

    module_cards = "\n".join(_module_card(m) for m in modules)

    summary_cards = "\n".join([
        _summary_card("Modules", str(total_modules)),
        _summary_card("Lines", f"{total_lines:,}"),
        _summary_card("Functions", str(total_functions)),
        _summary_card("Classes", str(total_classes)),
        _summary_card("Tests", str(total_tests)),
    ])

    signal_cards = _signal_summary_cards(signals_summary)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build comparison section if multiple packages
    comparison_html = ""
    all_packages = [metadata] + (compare_with or [])
    if len(all_packages) > 1:
        comparison_html = _comparison_section(all_packages)

    # Build additional package sections
    other_sections = ""
    for other in (compare_with or []):
        other_pkg = _escape_html(other.get("package", "unknown"))
        other_modules = other.get("modules", [])
        other_cards = "\n".join(_module_card(m) for m in other_modules)
        other_total_m = other.get("total_modules", 0)

        other_sections += f"""
    <div class="section">
        <div class="section-title">{other_pkg} — Module Map ({other_total_m} modules)</div>
        <div class="module-grid">
            {other_cards}
        </div>
    </div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{pkg} // HEALTH DASHBOARD</title>
<style>
/* ============================================================
   LIGHT TERMINAL AESTHETIC — inspired by personal-index
   Background: #fafbfc | Accent: #2d7d46 | Text: #1a1a2e
   Monospace font, grid cards, section headers
   ============================================================ */

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

html, body {{
    background: #fafbfc;
    color: #1a1a2e;
    font-family: 'SF Mono', 'Fira Code', 'Courier New', Courier, monospace;
    font-size: 15px;
    line-height: 1.6;
    min-height: 100vh;
}}

.dashboard {{
    max-width: 1300px;
    margin: 0 auto;
    padding: 2rem 2rem;
}}

/* ---- HEADER ---- */
.header {{
    border-bottom: 2px solid #2d7d46;
    padding-bottom: 1.2rem;
    margin-bottom: 2.5rem;
}}

.header h1 {{
    font-size: 1.8rem;
    font-weight: 400;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2d7d46;
}}

.header .prompt {{
    color: #5a7a6a;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}}

.header .prompt::before {{
    content: "> ";
    color: #2d7d46;
}}

/* ---- SECTION HEADERS ---- */
.section {{
    margin-bottom: 3rem;
}}

.section-title {{
    font-size: 1.1rem;
    font-weight: 400;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2d7d46;
    border-bottom: 1px solid #d0e0d8;
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}}

.section-title::before {{
    content: "## ";
    color: #5a7a6a;
}}

/* ---- SURFACE STATS ---- */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.8rem;
}}

.stat-card {{
    border: 1px solid #d0e0d8;
    padding: 0.9rem;
    text-align: center;
    background: #fff;
    transition: border-color 0.2s;
}}

.stat-card:hover {{
    border-color: #2d7d46;
}}

.stat-value {{
    font-size: 2rem;
    color: #2d7d46;
    font-weight: 400;
}}

.stat-label {{
    font-size: 0.8rem;
    color: #5a7a6a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}}

.stat-card.warn .stat-value {{
    color: #c47a20;
}}

.stat-card.err .stat-value {{
    color: #c0392b;
}}

/* ---- SIGNAL SUMMARY ROW ---- */
.signal-summary {{
    display: flex;
    gap: 1.2rem;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
    color: #5a7a6a;
}}

.signal-summary span {{
    color: #2d7d46;
    font-weight: 600;
}}

/* ---- COMPARISON TABLE ---- */
.comp-table-wrapper {{
    overflow-x: auto;
    border: 1px solid #d0e0d8;
    background: #fff;
}}

.comp-table {{
    width: 100%;
    border-collapse: collapse;
}}

.comp-table th {{
    background: #f4f9f6;
    padding: 0.8rem 1rem;
    text-align: center;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #2d7d46;
    border-bottom: 2px solid #d0e0d8;
}}

.comp-table th.comp-label {{
    text-align: left;
}}

.comp-table td {{
    padding: 0.6rem 1rem;
    border-bottom: 1px solid #f0f4f2;
    font-size: 0.95rem;
}}

.comp-table td.comp-label {{
    color: #5a7a6a;
    font-weight: 400;
}}

.comp-table td.comp-val {{
    text-align: center;
    color: #1a1a2e;
}}

.comp-table tr:hover {{
    background: #f9fbfa;
}}

/* ---- MODULE GRID ---- */
.module-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 0.6rem;
}}

.mod-card {{
    border: 1px solid #d0e0d8;
    padding: 0.65rem 0.85rem;
    background: #fff;
    transition: background 0.2s, border-color 0.2s;
}}

.mod-card:hover {{
    background: #f4f9f6;
    border-color: #2d7d46;
}}

.mod-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.3rem;
    gap: 0.6rem;
}}

.mod-name {{
    font-size: 0.95rem;
    color: #2d7d46;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
}}

.mod-status {{
    font-size: 0.75rem;
    font-weight: 400;
    letter-spacing: 1px;
    white-space: nowrap;
}}

.mod-meta {{
    display: flex;
    gap: 0.8rem;
    font-size: 0.78rem;
    color: #5a7a6a;
    flex-wrap: wrap;
}}

.mod-meta span::before {{
    color: #8aab9a;
}}

/* ---- SIGNAL BADGES ---- */
.badge {{
    display: inline-block;
    padding: 0.15em 0.5em;
    border-radius: 2px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 0.1em 0.1em;
    white-space: nowrap;
}}

.badge-s1 {{ background: #fff3cd; color: #856404; }}
.badge-s2 {{ background: #ffecb5; color: #6d5a00; }}
.badge-s3 {{ background: #e8e8e8; color: #383d41; }}
.badge-s4 {{ background: #f8d7da; color: #721c24; }}

.clear {{ color: #2d7d46; font-weight: 600; font-size: 0.82rem; }}

/* ---- CONTROLS ---- */
.controls {{
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}}

.controls input, .controls select {{
    padding: 0.55rem 0.75rem;
    border: 1px solid #c0d0c8;
    border-radius: 2px;
    font-family: inherit;
    font-size: 0.95rem;
    outline: none;
    background: #fff;
}}

.controls input:focus, .controls select:focus {{
    border-color: #2d7d46;
}}

.controls input {{ flex: 1; min-width: 220px; }}

/* ---- FOOTER ---- */
.footer {{
    margin-top: 2.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #d0e0d8;
    color: #8aab9a;
    font-size: 0.82rem;
    text-align: center;
    letter-spacing: 1px;
}}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: #fafbfc; }}
::-webkit-scrollbar-thumb {{ background: #c0d0c8; }}
::-webkit-scrollbar-thumb:hover {{ background: #8aab9a; }}

/* ---- RESPONSIVE ---- */
@media (max-width: 768px) {{
    .dashboard {{ padding: 1rem; }}
    .header h1 {{ font-size: 1.3rem; }}
    .stats-grid {{ grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.5rem; }}
    .module-grid {{ grid-template-columns: 1fr; }}
    .signal-summary {{ flex-wrap: wrap; gap: 0.8rem; }}
}}
</style>
</head>
<body>
<div class="dashboard">

<!-- HEADER -->
<div class="header">
    <h1>{pkg} // HEALTH DASHBOARD</h1>
    <div class="prompt">codebase projection — {total_modules} modules scanned · {now}</div>
</div>

<!-- 1. SURFACE STATS -->
<div class="section">
    <div class="section-title">Surface Stats</div>
    <div class="stats-grid">
        {summary_cards}
    </div>
</div>

<!-- 2. SIGNAL SUMMARY -->
<div class="section">
    <div class="section-title">Health Signals</div>
    <div class="stats-grid">
        {signal_cards}
    </div>
    <div class="signal-summary">
        <div>S1 No Tests: <span>{signals_summary.get("S1", 0)}</span></div>
        <div>S2 Oversized: <span>{signals_summary.get("S2", 0)}</span></div>
        <div>S3 Dead Code: <span>{signals_summary.get("S3", 0)}</span></div>
        <div>S4 Lint Errors: <span>{signals_summary.get("S4", 0)}</span></div>
    </div>
</div>

<!-- 3. COMPARISON (if multiple packages) -->
{comparison_html}

<!-- 4. MODULE MAP -->
<div class="section">
    <div class="section-title">{pkg} — Module Map ({total_modules} modules)</div>
    <div class="controls">
        <input type="text" id="search" placeholder="Search modules..." oninput="filterModules()">
        <select id="signalFilter" onchange="filterModules()">
            <option value="all">All Status</option>
            <option value="S1">S1 — No Tests</option>
            <option value="S2">S2 — Oversized</option>
            <option value="S3">S3 — Dead Code</option>
            <option value="S4">S4 — Lint Errors</option>
            <option value="clear">Clear Only</option>
        </select>
    </div>
    <div class="module-grid" id="moduleGrid">
        {module_cards}
    </div>
</div>

{other_sections}

<!-- FOOTER -->
<div class="footer">
    {pkg} health dashboard · {total_modules} modules · {total_lines:,} lines · {total_functions} functions
</div>
</div>

<script>
function filterModules() {{
    const query = document.getElementById("search").value.toLowerCase();
    const signalFilter = document.getElementById("signalFilter").value;
    const cards = document.querySelectorAll("#moduleGrid .mod-card");

    cards.forEach(card => {{
        const name = card.querySelector(".mod-name").textContent.toLowerCase();
        const signals = card.querySelector(".mod-signals").textContent;
        const matchesSearch = name.includes(query);

        let matchesSignal = true;
        if (signalFilter === "clear") {{
            matchesSignal = signals.includes("clear");
        }} else if (signalFilter !== "all") {{
            matchesSignal = signals.includes(signalFilter);
        }}

        card.style.display = (matchesSearch && matchesSignal) ? "" : "none";
    }});
}}
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Publish helper
# ---------------------------------------------------------------------------


def publish(
    html: str,
    output_path: str | Path = "dashboard.html",
    sync: bool = False,
) -> Path:
    """Write HTML to *output_path* and optionally sync to ``gh-pages``."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("Dashboard HTML written to %s", out)

    if sync:
        _sync_to_ghpages(out)

    return out


# ---------------------------------------------------------------------------
# GitHub Pages sync
# ---------------------------------------------------------------------------


def _sync_to_ghpages(html_path: Path) -> None:
    """Push *html_path* to the ``gh-pages`` branch."""
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"git not available: {exc}") from exc

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        repo_root = Path(result.stdout.strip())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Not a git repository: {exc}") from exc

    gh_pages_dir = repo_root / ".gh-pages-staging"
    gh_pages_dir.mkdir(exist_ok=True)

    dest = gh_pages_dir / html_path.name
    dest.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")

    steps = [
        ["git", "-C", str(repo_root), "checkout", "gh-pages"],
        ["git", "-C", str(repo_root), "cp", str(dest), str(repo_root / html_path.name)],
        ["git", "-C", str(repo_root), "add", html_path.name],
        ["git", "-C", str(repo_root), "commit", "-m",
         f"docs: update dashboard — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"],
        ["git", "-C", str(repo_root), "push", "origin", "gh-pages"],
    ]

    for step in steps:
        cmd_str = " ".join(step)
        logger.debug("Running: %s", cmd_str)
        result = subprocess.run(step, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            if "nothing to commit" in result.stderr.lower():
                logger.info("No changes to commit, skipping")
                break
            logger.warning("Command failed: %s\nstderr: %s", cmd_str, result.stderr)

    try:
        import shutil
        shutil.rmtree(gh_pages_dir)
    except OSError:
        pass

    logger.info("Dashboard synced to gh-pages")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for publish_dashboard."""
    parser = argparse.ArgumentParser(
        description="Generate HTML dashboard from alloc metadata"
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=str,
        default=None,
        help="Path to pre-generated dashboard JSON",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        default=None,
        help="Paths to additional JSON files for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dashboard.html",
        help="Output HTML file path (default: dashboard.html)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Push generated HTML to gh-pages branch",
    )
    args = parser.parse_args()

    # Load primary metadata
    if args.json_path:
        json_path = Path(args.json_path)
        if not json_path.exists():
            logger.error("JSON file not found: %s", json_path)
            sys.exit(1)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        from alloc.lib.dashboard import generate_json as _gen_json

        json_str = _gen_json()
        metadata = json.loads(json_str)

    # Load comparison metadata
    compare_with = []
    if args.compare:
        for path_str in args.compare:
            p = Path(path_str)
            if p.exists():
                compare_with.append(json.loads(p.read_text(encoding="utf-8")))
            else:
                logger.warning("Comparison file not found: %s", p)

    # Generate and publish
    html = generate_html(metadata, compare_with=compare_with if compare_with else None)
    publish(html, output_path=args.output, sync=args.sync)
    print(f"Dashboard published to {args.output}")


if __name__ == "__main__":
    main()
