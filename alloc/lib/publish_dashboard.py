"""alloc.lib.publish_dashboard — HTML publisher for dashboard metadata.

Takes the JSON metadata produced by :mod:`alloc.lib.dashboard` and renders
a standalone, responsive HTML page with inline CSS/JS.  Signals S1–S4 are
colour-coded by severity.  An optional ``--sync`` CLI flag pushes the
generated HTML to a ``gh-pages`` branch for GitHub Pages hosting.

Usage
-----
    from alloc.lib.publish_dashboard import generate_html, publish

    html = generate_html(metadata_dict)
    publish(html, output_path="dashboard.html", sync=False)

Or from CLI::

    python -m alloc.lib.publish_dashboard [--json PATH] [--output PATH] [--sync]

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
    """Return an HTML span for a single signal badge.

    Parameters
    ----------
    signal : str
        Signal string like ``"S1:no_tests"`` or ``"S4:lint_errors(3)"``.

    Returns
    -------
    str
        HTML ``<span>`` element with colour-coded styling.
    """
    key = signal.split(":")[0]
    label = signal.split(":", 1)[1] if ":" in signal else key
    bg, fg = SIGNAL_COLORS.get(key, ("#e9ecef", "#495057"))
    icon = SIGNAL_ICONS.get(key, "⚠️")
    return (
        f'<span class="badge badge-{key.lower()}" '
        f'style="background:{bg};color:{fg}">'
        f"{icon} {_escape_html(label)}</span>"
    )


def _module_row(mod: dict[str, Any]) -> str:
    """Return an HTML table row for a single module.

    Parameters
    ----------
    mod : dict
        One module entry from the dashboard metadata.

    Returns
    -------
    str
        HTML ``<tr>`` element.
    """
    path = _escape_html(mod.get("path", "?"))
    lines = mod.get("lines", 0)
    functions = mod.get("functions", 0)
    classes = mod.get("classes", 0)
    tests = mod.get("test_count", 0)
    has_doc = mod.get("has_docstring", False)
    signals = mod.get("signals", [])

    # Docstring indicator
    doc_icon = "✅" if has_doc else "❌"

    # Test coverage bar
    test_pct = min(tests * 10, 100) if tests > 0 else 0
    test_bar = (
        f'<div class="test-bar">'
        f'<div class="test-bar-fill" style="width:{test_pct}%"></div>'
        f"</div>"
    )

    # Signal badges
    badges_html = " ".join(_signal_badge(s) for s in signals)
    signals_cell = badges_html if badges_html else '<span class="clear">✅ clear</span>'

    return (
        f"<tr>"
        f"<td class=\"mod-path\">{path}</td>"
        f"<td class=\"mod-stat\">{lines}</td>"
        f"<td class=\"mod-stat\">{functions}</td>"
        f"<td class=\"mod-stat\">{classes}</td>"
        f"<td class=\"mod-stat\">{doc_icon}</td>"
        f"<td class=\"mod-tests\">{tests} {test_bar}</td>"
        f"<td class=\"mod-signals\">{signals_cell}</td>"
        f"</tr>"
    )


def _summary_card(title: str, value: str, icon: str = "") -> str:
    """Return an HTML summary card div.

    Parameters
    ----------
    title : str
        Card title (e.g. "Modules").
    value : str
        Card value (e.g. "18").
    icon : str
        Optional emoji icon.

    Returns
    -------
    str
        HTML ``<div>`` card element.
    """
    return (
        f'<div class="summary-card">'
        f"<div class=\"card-icon\">{icon}</div>"
        f"<div class=\"card-value\">{_escape_html(value)}</div>"
        f"<div class=\"card-title\">{_escape_html(title)}</div>"
        f"</div>"
    )


def _signal_summary_cards(signals_summary: dict[str, int]) -> str:
    """Return HTML cards for each signal type count.

    Parameters
    ----------
    signals_summary : dict
        Mapping of signal key (S1-S4) to count.

    Returns
    -------
    str
        HTML fragment with signal summary cards.
    """
    if not signals_summary:
        return '<div class="summary-card"><div class="card-value">0</div>' \
               '<div class="card-title">All Clear ✅</div></div>'

    cards = []
    for key in sorted(signals_summary.keys()):
        count = signals_summary[key]
        bg, fg = SIGNAL_COLORS.get(key, ("#e9ecef", "#495057"))
        icon = SIGNAL_ICONS.get(key, "⚠️")
        label = SIGNAL_LABELS.get(key, key)
        cards.append(
            f'<div class="summary-card signal-card" '
            f'style="background:{bg};color:{fg}">'
            f"<div class=\"card-icon\">{icon}</div>"
            f"<div class=\"card-value\">{count}</div>"
            f"<div class=\"card-title\">{label}</div>"
            f"</div>"
        )
    return "\n".join(cards)


def generate_html(metadata: dict[str, Any]) -> str:
    """Generate a standalone HTML dashboard page from metadata.

    The output is a self-contained HTML document with inline CSS and JS —
    no external dependencies required.  Works on mobile and desktop via
    responsive CSS Grid/Flexbox layout.

    Parameters
    ----------
    metadata : dict
        The deserialised JSON output of :func:`dashboard.generate_json`.
        Must contain keys: ``package``, ``total_modules``, ``total_lines``,
        ``total_functions``, ``total_classes``, ``total_tests``,
        ``signals_summary``, ``modules``.

    Returns
    -------
    str
        Complete HTML document string.
    """
    pkg = _escape_html(metadata.get("package", "unknown"))
    total_modules = metadata.get("total_modules", 0)
    total_lines = metadata.get("total_lines", 0)
    total_functions = metadata.get("total_functions", 0)
    total_classes = metadata.get("total_classes", 0)
    total_tests = metadata.get("total_tests", 0)
    signals_summary = metadata.get("signals_summary", {})
    modules = metadata.get("modules", [])

    # Module rows
    module_rows = "\n".join(_module_row(m) for m in modules)

    # Summary cards
    summary_cards = "\n".join([
        _summary_card("Modules", str(total_modules), "📁"),
        _summary_card("Lines", f"{total_lines:,}", "📝"),
        _summary_card("Functions", str(total_functions), "⚙️"),
        _summary_card("Classes", str(total_classes), "🏗️"),
        _summary_card("Tests", str(total_tests), "🧪"),
    ])

    # Signal summary
    signal_cards = _signal_summary_cards(signals_summary)

    # Timestamp
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{pkg} — Health Dashboard</title>
<style>
/* ── Reset & Base ───────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: #f5f7fa;
    color: #212529;
    line-height: 1.6;
    padding: 1rem;
}}

/* ── Layout ─────────────────────────────────────────── */
.dashboard {{
    max-width: 1200px;
    margin: 0 auto;
}}
.header {{
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}}
.header h1 {{ font-size: 1.75rem; font-weight: 700; }}
.header .timestamp {{ font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem; }}

/* ── Summary Cards ──────────────────────────────────── */
.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}}
.summary-card {{
    background: white;
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: transform 0.15s ease;
}}
.summary-card:hover {{ transform: translateY(-2px); }}
.summary-card .card-icon {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
.summary-card .card-value {{ font-size: 1.75rem; font-weight: 700; }}
.summary-card .card-title {{ font-size: 0.8rem; text-transform: uppercase;
.signal-card {{ font-weight: 600; }}

/* ── Signal Legend ──────────────────────────────────── */
.signal-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
    padding: 1rem;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.signal-legend h3 {{ width: 100%; font-size: 0.9rem; margin-bottom: 0.5rem; }}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.85rem;
}}

/* ── Module Table ───────────────────────────────────── */
.table-wrapper {{
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    overflow-x: auto;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}
thead th {{
    background: #f8f9fa;
    padding: 0.75rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid #e9ecef;
    position: sticky;
    top: 0;
}}
tbody tr {{ border-bottom: 1px solid #f0f0f0; }}
tbody tr:hover {{ background: #f8f9fa; }}
td {{ padding: 0.65rem 1rem; vertical-align: middle; }}
.mod-path {{ font-family: "SF Mono", "Fira Code", monospace; font-size: 0.85rem; color: #495057; }}
.mod-stat {{ text-align: center; font-variant-numeric: tabular-nums; }}
.mod-tests {{ min-width: 100px; }}

/* ── Test Bar ───────────────────────────────────────── */
.test-bar {{
    height: 6px;
    background: #e9ecef;
    border-radius: 3px;
    margin-top: 4px;
    overflow: hidden;
}}
.test-bar-fill {{
    height: 100%;
    background: #28a745;
    border-radius: 3px;
    transition: width 0.3s ease;
}}

/* ── Badges ─────────────────────────────────────────── */
.badge {{
    display: inline-block;
    padding: 0.2em 0.6em;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0.15em;
    white-space: nowrap;
}}
.clear {{ color: #28a745; font-weight: 600; font-size: 0.85rem; }}

/* ── Search / Filter ────────────────────────────────── */
.controls {{
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}
.controls input, .controls select {{
    padding: 0.5rem 0.75rem;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    font-size: 0.9rem;
    outline: none;
}}
.controls input:focus, .controls select:focus {{
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.15);
}}
.controls input {{ flex: 1; min-width: 200px; }}

/* ── Footer ─────────────────────────────────────────── */
.footer {{
    text-align: center;
    padding: 1.5rem;
    font-size: 0.8rem;
    color: #868e96;
}}

/* ── Responsive ─────────────────────────────────────── */
@media (max-width: 768px) {{
    body {{ padding: 0.5rem; }}
    .header h1 {{ font-size: 1.3rem; }}
    .summary-grid {{ grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 0.5rem; }}
    .summary-card {{ padding: 0.75rem; }}
    .summary-card .card-value {{ font-size: 1.3rem; }}
    thead th {{ font-size: 0.7rem; padding: 0.5rem; }}
    td {{ padding: 0.5rem; font-size: 0.8rem; }}
    .mod-path {{ font-size: 0.75rem; word-break: break-all; }}
}}

@media (max-width: 480px) {{
    .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .controls {{ flex-direction: column; }}
    .controls input {{ min-width: unset; }}
}}
</style>
</head>
<body>
<div class="dashboard">
    <!-- Header -->
    <div class="header">
        <h1>📊 {pkg} Health Dashboard</h1>
        <div class="timestamp">Generated: {now}</div>
    </div>

    <!-- Summary Cards -->
    <div class="summary-grid">
        {summary_cards}
    </div>

    <!-- Signal Summary -->
    <div class="summary-grid">
        {signal_cards}
    </div>

    <!-- Signal Legend -->
    <div class="signal-legend">
        <h3>Signal Legend</h3>
        <div class="legend-item">🧪 <strong>S1</strong> — No Tests</div>
        <div class="legend-item">📦 <strong>S2</strong> — Oversized
            (&gt;200 lines &amp; &gt;15 funcs)</div>
        <div class="legend-item">💀 <strong>S3</strong> — Dead Code (0 internal imports)</div>
        <div class="legend-item">🔧 <strong>S4</strong> — Lint/Type Errors</div>
    </div>

    <!-- Controls -->
    <div class="controls">
        <input type="text" id="search" placeholder="Search modules..." oninput="filterTable()">
        <select id="signalFilter" onchange="filterTable()">
            <option value="all">All Signals</option>
            <option value="S1">S1 — No Tests</option>
            <option value="S2">S2 — Oversized</option>
            <option value="S3">S3 — Dead Code</option>
            <option value="S4">S4 — Lint Errors</option>
            <option value="clear">✅ Clear Only</option>
        </select>
    </div>

    <!-- Module Table -->
    <div class="table-wrapper">
        <table id="moduleTable">
            <thead>
                <tr>
                    <th>Module</th>
                    <th>Lines</th>
                    <th>Funcs</th>
                    <th>Classes</th>
                    <th>Doc</th>
                    <th>Tests</th>
                    <th>Signals</th>
                </tr>
            </thead>
            <tbody>
                {module_rows}
            </tbody>
        </table>
    </div>

    <!-- Footer -->
    <div class="footer">
        alloc health dashboard · {total_modules} modules ·
        {total_lines:,} lines · {total_functions} functions
    </div>
</div>

<script>
function filterTable() {{
    const query = document.getElementById("search").value.toLowerCase();
    const signalFilter = document.getElementById("signalFilter").value;
    const rows = document.querySelectorAll("#moduleTable tbody tr");

    rows.forEach(row => {{
        const path = row.querySelector(".mod-path").textContent.toLowerCase();
        const signals = row.querySelector(".mod-signals").textContent;
        const matchesSearch = path.includes(query);

        let matchesSignal = true;
        if (signalFilter === "clear") {{
            matchesSignal = signals.includes("clear");
        }} else if (signalFilter !== "all") {{
            matchesSignal = signals.includes(signalFilter);
        }}

        row.style.display = (matchesSearch && matchesSignal) ? "" : "none";
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
    """Write HTML to *output_path* and optionally sync to ``gh-pages``.

    Parameters
    ----------
    html : str
        The HTML document string (from :func:`generate_html`).
    output_path : str or Path
        File path to write the HTML to.
    sync : bool
        If ``True``, push the HTML to a ``gh-pages`` branch via git.

    Returns
    -------
    Path
        The path where the HTML was written.
    """
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
    """Push *html_path* to the ``gh-pages`` branch.

    Uses git subcommands to:
    1. Create or checkout ``gh-pages`` branch
    2. Copy the HTML file to the branch root
    3. Commit and push

    Parameters
    ----------
    html_path : Path
        Path to the generated HTML file.

    Raises
    ------
    RuntimeError
        If git is not available or push fails.
    """
    try:
        # Check git is available
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"git not available: {exc}") from exc

    # Determine repo root
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

    # Copy HTML to staging
    dest = gh_pages_dir / html_path.name
    dest.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Stage, commit, push
    steps = [
        ["git", "-C", str(repo_root), "checkout", "gh-pages"],
        ["git", "-C", str(repo_root), "cp", str(dest), str(repo_root / html_path.name)],
        ["git", "-C", str(repo_root), "add", html_path.name],
        ["git", "-C", str(repo_root), "commit", "-m",
         f"docs: update dashboard — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"],
        ["git", "-C", str(repo_root), "push", "origin", "gh-pages"],
    ]

    # Handle "nothing to commit" gracefully
    for step in steps:
        cmd_str = " ".join(step)
        logger.debug("Running: %s", cmd_str)
        result = subprocess.run(step, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            # "nothing to commit" is acceptable
            if "nothing to commit" in result.stderr.lower():
                logger.info("No changes to commit, skipping")
                break
            logger.warning("Command failed: %s\nstderr: %s", cmd_str, result.stderr)

    # Clean up staging
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
    """CLI entry point for publish_dashboard.

    Usage::

        python -m alloc.lib.publish_dashboard [--json PATH] [--output PATH] [--sync]

    If ``--json`` is provided, reads metadata from that file.
    Otherwise generates fresh metadata via :mod:`alloc.lib.dashboard`.

    Options
    -------
    --json PATH
        Path to pre-generated dashboard JSON.
    --output PATH
        Output HTML file path (default: ``dashboard.html``).
    --sync
        Push generated HTML to ``gh-pages`` branch.
    """
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

    # Load or generate metadata
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

    # Generate and publish
    html = generate_html(metadata)
    publish(html, output_path=args.output, sync=args.sync)
    print(f"Dashboard published to {args.output}")


if __name__ == "__main__":
    main()
