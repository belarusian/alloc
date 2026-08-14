"""alloc.lib.dashboard — Codebase health dashboard.

Crawls the ``alloc/`` package to extract per-module statistics
(lines, functions, classes, test counts) and detects health signals:

* **S1** — module has no tests
* **S2** — module is oversized (>200 lines AND >15 functions)
* **S3** — dead code (module has 0 imports from other alloc modules)
* **S4** — lint/type errors (detected via ``ruff check``)

Generates JSON metadata consumed by :mod:`alloc.lib.cycle_signals`.
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModuleStats:
    """Statistics for a single Python module."""

    path: str
    lines: int
    functions: int
    classes: int
    has_docstring: bool
    imports: list[str] = field(default_factory=list)
    test_count: int = 0
    signals: list[str] = field(default_factory=list)


@dataclass
class DashboardMetadata:
    """Top-level dashboard metadata."""

    package: str
    total_modules: int
    total_lines: int
    total_functions: int
    total_classes: int
    total_tests: int
    modules: list[dict[str, Any]] = field(default_factory=list)
    signals_summary: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Crawling helpers
# ---------------------------------------------------------------------------


def _count_lines(filepath: Path) -> int:
    """Return the number of non-blank, non-comment lines in *filepath*."""
    count = 0
    try:
        text = filepath.read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


def _parse_ast(
    filepath: Path,
) -> tuple[int, int, bool, list[str]]:
    """Parse a Python file and return (functions, classes, has_docstring, imports).

    *functions* counts top-level and method ``def`` nodes.
    *classes* counts top-level ``class`` nodes.
    *imports* lists fully-qualified import names from within the ``alloc``
    package (e.g. ``alloc.config.settings``, ``alloc.lib.cache``).
    """
    try:
        source = filepath.read_text(encoding="utf-8")
    except OSError:
        return 0, 0, False, []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return 0, 0, False, []

    # Docstring check
    has_docstring = bool(ast.get_docstring(tree))

    functions = 0
    classes = 0
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(
            node, ast.AsyncFunctionDef
        ):
            functions += 1
        elif isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("alloc."):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("alloc."):
                imports.append(node.module)

    return functions, classes, has_docstring, imports


def _find_test_count(
    module_path: Path,
    tests_dir: Path,
    package_root: Path,
) -> int:
    """Count test functions/methods that reference *module_path*.

    Heuristic: look for test files whose name matches the module stem,
    then count ``def test_`` functions inside them.
    """
    # Derive expected test file name from module path
    rel = module_path.relative_to(package_root)
    parts = rel.with_suffix("").parts  # e.g. ('lib', 'cache')

    # Common test file naming conventions (use set to deduplicate)
    candidates: set[str] = set()
    if len(parts) == 1:
        # e.g. core.py -> test_core.py
        candidates.add(f"test_{parts[0]}.py")
    else:
        # e.g. lib/cache.py -> test_cache.py
        candidates.add(f"test_{parts[-1]}.py")
        # e.g. lib/cache.py -> test_lib_cache.py
        candidates.add(f"test_{'_'.join(parts)}.py")

    count = 0
    for candidate in sorted(candidates):
        test_file = tests_dir / candidate
        if test_file.exists():
            try:
                source = test_file.read_text(encoding="utf-8")
            except OSError:
                continue
            try:
                tree = ast.parse(source, filename=str(test_file))
            except SyntaxError:
                continue
            # Count test functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("test_"):
                        count += 1
    return count


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def _detect_s1_no_tests(stats: ModuleStats) -> bool:
    """S1: module has no associated tests."""
    return stats.test_count == 0


def _detect_s2_oversized(stats: ModuleStats) -> bool:
    """S2: module is oversized (>200 lines AND >15 functions)."""
    return stats.lines > 200 and stats.functions > 15


def _detect_s3_dead_code(
    stats: ModuleStats,
    all_imports: dict[str, list[str]],
) -> bool:
    """S3: dead code — no other alloc module imports this module.

    A module is considered dead if its fully-qualified name never appears
    in the import lists of any other module.
    """
    module_name = stats.path  # e.g. "alloc.lib.cache"
    for other_name, other_imports in all_imports.items():
        if other_name == module_name:
            continue
        for imp in other_imports:
            if imp == module_name or imp.startswith(module_name + "."):
                return False  # imported somewhere
    return True


def _detect_s4_lint_errors(filepath: Path) -> list[str]:
    """S4: run ``ruff check`` on *filepath* and return error messages."""
    try:
        result = subprocess.run(
            ["ruff", "check", str(filepath), "--output-format", "concise"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = []
        for line in result.stdout.strip().splitlines():
            if line and line != str(filepath):
                lines.append(line.strip())
        for line in result.stderr.strip().splitlines():
            if line:
                lines.append(line.strip())
        return lines
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def crawl_package(
    package_path: str | Path = "alloc",
    tests_path: str | Path = "tests",
) -> DashboardMetadata:
    """Crawl the *package_path* and return a :class:`DashboardMetadata`.

    Parameters
    ----------
    package_path : str or Path
        Root directory of the Python package (default ``"alloc"``).
    tests_path : str or Path
        Directory containing test files (default ``"tests"``).

    Returns
    -------
    DashboardMetadata
        Complete metadata with per-module stats and signal flags.
    """
    pkg = Path(package_path)
    tests_dir = Path(tests_path)

    if not pkg.exists():
        raise FileNotFoundError(f"Package path does not exist: {pkg}")

    # Collect all .py files (skip __pycache__)
    py_files = sorted(
        p for p in pkg.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    )

    modules: list[ModuleStats] = []
    all_imports: dict[str, list[str]] = {}

    for py_file in py_files:
        # Build fully-qualified module name
        rel = py_file.relative_to(pkg.parent)
        parts = rel.with_suffix("").parts
        module_name = ".".join(parts)  # e.g. "alloc.lib.cache"

        lines = _count_lines(py_file)
        functions, classes, has_docstring, imports = _parse_ast(py_file)
        test_count = _find_test_count(py_file, tests_dir, pkg)

        stats = ModuleStats(
            path=module_name,
            lines=lines,
            functions=functions,
            classes=classes,
            has_docstring=has_docstring,
            imports=imports,
            test_count=test_count,
        )
        all_imports[module_name] = imports
        modules.append(stats)

    # Detect signals (need full import map for S3)
    for stats in modules:
        if _detect_s1_no_tests(stats):
            stats.signals.append("S1:no_tests")
        if _detect_s2_oversized(stats):
            stats.signals.append("S2:oversized")
        if _detect_s3_dead_code(stats, all_imports):
            stats.signals.append("S3:dead_code")

        # Resolve actual file path for lint check
        file_path = Path(
            str(pkg.parent) + "/" + stats.path.replace(".", "/") + ".py"
        )
        lint_errors = _detect_s4_lint_errors(file_path)
        if lint_errors:
            stats.signals.append(f"S4:lint_errors({len(lint_errors)})")

    # Aggregate
    total_lines = sum(m.lines for m in modules)
    total_functions = sum(m.functions for m in modules)
    total_classes = sum(m.classes for m in modules)
    total_tests = sum(m.test_count for m in modules)

    signals_summary: dict[str, int] = {}
    for m in modules:
        for sig in m.signals:
            key = sig.split(":")[0]  # S1, S2, S3, S4
            signals_summary[key] = signals_summary.get(key, 0) + 1

    metadata = DashboardMetadata(
        package=str(pkg),
        total_modules=len(modules),
        total_lines=total_lines,
        total_functions=total_functions,
        total_classes=total_classes,
        total_tests=total_tests,
        modules=[asdict(m) for m in modules],
        signals_summary=signals_summary,
    )

    return metadata


def generate_json(
    package_path: str | Path = "alloc",
    tests_path: str | Path = "tests",
    output_path: str | Path | None = None,
) -> str:
    """Crawl the package and return JSON string.

    If *output_path* is given, also write the JSON to that file.

    Parameters
    ----------
    package_path : str or Path
        Root directory of the Python package.
    tests_path : str or Path
        Directory containing test files.
    output_path : str or Path, optional
        File path to write JSON output.

    Returns
    -------
    str
        JSON-serialised metadata.
    """
    metadata = crawl_package(package_path, tests_path)
    json_str = json.dumps(asdict(metadata), indent=2, default=str)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_str, encoding="utf-8")
        logger.info("Dashboard metadata written to %s", out)

    return json_str
