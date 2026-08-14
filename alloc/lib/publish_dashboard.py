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


def _module_card(mod: dict[str, Any]) -> str:
    """Return an HTML module card div.

    Parameters
    ----------
    mod : dict
        One module entry from the dashboard metadata.

    Returns
    -------
    str
        HTML ``<div>`` card element.
    """
    path = _escape_html(mod.get("path", "?"))
    lines = mod.get("lines", 0)
    functions = mod.get("functions", 0)
    classes = mod.get("classes", 0)
    tests = mod.get("test_count", 0)
    signals = mod.get("signals", [])

    # Signal badges
    badges_html = " ".join(_signal_badge(s) for s in signals)
    if badges_html:
        status_class = "mod-signals"
        status_html = badges_html
    else:
        status_class = "clear"
        status_html = "✅ clear"

    # Border color based on signals
    border_color = "#22c55e" if not signals else "#c0392b"

    return (
        f'<div class="mod-card" style="border-color: {border_color};">'
        f'<div class="mod-header">'
        f'<span class="mod-name">{path}</span>'
        f'<span class="{status_class}">{status_html}</span>'
        f'</div>'
        f'<div class="mod-meta">'
        f'<span>CLASSES: {classes}</span>'
        f'<span>FUNCS: {functions}</span>'
        f'<span>LINES: {lines}</span>'
        f'<span>TESTS: {tests}</span>'
        f'</div>'
        f'</div>'
    )


def _summary_card(title: str, value: str, icon: str = "", card_class: str = "") -> str:
    """Return an HTML stat card div.

    Parameters
    ----------
    title : str
        Card title (e.g. "Modules").
    value : str
        Card value (e.g. "18").
    icon : str
        Optional emoji icon.
    card_class : str
        Additional CSS class (e.g. "warn", "err").

    Returns
    -------
    str
        HTML ``<div>`` card element.
    """
    cls = f"stat-card {card_class}" if card_class else "stat-card"
    return (
        f'<div class="{cls}">'
        f"<div class=\"stat-value\">{_escape_html(value)}</div>"
        f"<div class=\"stat-label\">{_escape_html(title)}</div>"
        f"</div>"
    )


def _signal_summary_cards(signals_summary: dict[str, int]) -> str:
    """Return HTML stat cards for each signal type count.

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

    # Module cards (grid layout, inspired by personal-index)
    module_cards = "\n".join(_module_card(m) for m in modules)

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
    font-size: 13px;
    line-height: 1.5;
    min-height: 100vh;
}}

.dashboard {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
}}

/* ---- HEADER ---- */
.header {{
    border-bottom: 2px solid #2d7d46;
    padding-bottom: 1rem;
    margin-bottom: 2rem;
}}

.header h1 {{
    font-size: 1.4rem;
    font-weight: 400;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2d7d46;
}}

.header .prompt {{
    color: #5a7a6a;
    font-size: 0.8rem;
    margin-top: 0.4rem;
}}

.header .prompt::before {{
    content: "> ";
    color: #2d7d46;
}}

/* ---- SECTION HEADERS ---- */
.section {{
    margin-bottom: 2.5rem;
}}

.section-title {{
    font-size: 0.9rem;
    font-weight: 400;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #2d7d46;
    border-bottom: 1px solid #d0e0d8;
    padding-bottom: 0.35rem;
    margin-bottom: 1rem;
}}

.section-title::before {{
    content: "## ";
    color: #5a7a6a;
}}

/* ---- SURFACE STATS ---- */
.stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 0.65rem;
}}

.stat-card {{
    border: 1px solid #d0e0d8;
    padding: 0.7rem;
    text-align: center;
    background: #fff;
    transition: border-color 0.2s;
}}

.stat-card:hover {{
    border-color: #2d7d46;
}}

.stat-value {{
    font-size: 1.6rem;
    color: #2d7d46;
    font-weight: 400;
}}

.stat-label {{
    font-size: 0.65rem;
    color: #5a7a6a;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
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
    gap: 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.75rem;
    color: #5a7a6a;
}}

.signal-summary span {{
    color: #2d7d46;
    font-weight: 600;
}}

/* ---- MODULE GRID ---- */
.module-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 0.55rem;
}}

.mod-card {{
    border: 1px solid #d0e0d8;
    padding: 0.55rem 0.75rem;
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
    margin-bottom: 0.25rem;
    gap: 0.5rem;
}}

.mod-name {{
    font-size: 0.82rem;
    color: #2d7d46;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
}}

.mod-status {{
    font-size: 0.65rem;
    font-weight: 400;
    letter-spacing: 1px;
    white-space: nowrap;
}}

.mod-meta {{
    display: flex;
    gap: 0.7rem;
    font-size: 0.65rem;
    color: #5a7a6a;
    flex-wrap: wrap;
}}

.mod-meta span::before {{
    color: #8aab9a;
}}

/* ---- SIGNAL BADGES ---- */
.badge {{
    display: inline-block;
    padding: 0.1em 0.45em;
    border-radius: 2px;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 0.1em 0.1em;
    white-space: nowrap;
}}

.badge-s1 {{ background: #fff3cd; color: #856404; }}
.badge-s2 {{ background: #ffecb5; color: #6d5a00; }}
.badge-s3 {{ background: #e8e8e8; color: #383d41; }}
.badge-s4 {{ background: #f8d7da; color: #721c24; }}

.clear {{ color: #2d7d46; font-weight: 600; font-size: 0.72rem; }}

/* ---- CONTROLS ---- */
.controls {{
    display: flex;
    gap: 0.65rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}}

.controls input, .controls select {{
    padding: 0.45rem 0.65rem;
    border: 1px solid #c0d0c8;
    border-radius: 2px;
    font-family: inherit;
    font-size: 0.82rem;
    outline: none;
    background: #fff;
}}

.controls input:focus, .controls select:focus {{
    border-color: #2d7d46;
}}

.controls input {{ flex: 1; min-width: 200px; }}

/* ---- FOOTER ---- */
.footer {{
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #d0e0d8;
    color: #8aab9a;
    font-size: 0.7rem;
    text-align: center;
    letter-spacing: 1px;
}}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: #fafbfc; }}
::-webkit-scrollbar-thumb {{ background: #c0d0c8; }}
::-webkit-scrollbar-thumb:hover {{ background: #8aab9a; }}

/* ---- RESPONSIVE ---- */
@media (max-width: 768px) {{
    .dashboard {{ padding: 1rem; }}
    .header h1 {{ font-size: 1.1rem; }}
    .stats-grid {{ grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 0.4rem; }}
    .module-grid {{ grid-template-columns: 1fr; }}
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
    {signal_cards}
    <div class="signal-summary">
        <div>🧪 S1 — No Tests: <span>{signals_summary.get("S1", 0)}</span></div>
        <div>📦 S2 — Oversized: <span>{signals_summary.get("S2", 0)}</span></div>
        <div>💀 S3 — Dead Code: <span>{signals_summary.get("S3", 0)}</span></div>
        <div>🔧 S4 — Lint Errors: <span>{signals_summary.get("S4", 0)}</span></div>
    </div>
</div>

<!-- 3. CONTROLS -->
<div class="section">
    <div class="section-title">Module Map</div>
    <div class="controls">
        <input type="text" id="search" placeholder="Search modules..." oninput="filterModules()">
        <select id="signalFilter" onchange="filterModules()">
            <option value="all">All Status</option>
            <option value="S1">S1 — No Tests</option>
            <option value="S2">S2 — Oversized</option>
            <option value="S3">S3 — Dead Code</option>
            <option value="S4">S4 — Lint Errors</option>
            <option value="clear">✅ Clear Only</option>
        </select>
    </div>
    <div class="module-grid" id="moduleGrid">
        {module_cards}
    </div>
</div>

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
