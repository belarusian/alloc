"""Tests for alloc.lib.dashboard — codebase health dashboard."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from alloc.lib.dashboard import (
    ModuleStats,
    DashboardMetadata,
    _count_lines,
    _parse_ast,
    _find_test_count,
    _detect_s1_no_tests,
    _detect_s2_oversized,
    _detect_s3_dead_code,
    _detect_s4_lint_errors,
    crawl_package,
    generate_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_pkg(tmp_path: Path) -> Path:
    """Create a minimal fake package under tmp_path."""
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""fake package."""\n')
    return pkg


@pytest.fixture
def tmp_tests(tmp_path: Path) -> Path:
    """Create a minimal tests directory under tmp_path."""
    td = tmp_path / "tests"
    td.mkdir()
    return td


# ---------------------------------------------------------------------------
# _count_lines
# ---------------------------------------------------------------------------


class TestCountLines:
    def test_counts_non_blank_non_comment(self, tmp_path: Path) -> None:
        f = tmp_path / "x.py"
        f.write_text(
            textwrap.dedent(
                """\
                # comment
                x = 1

                # another comment
                y = 2
                """
            )
        )
        assert _count_lines(f) == 2

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        assert _count_lines(f) == 0

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert _count_lines(tmp_path / "nope.py") == 0


# ---------------------------------------------------------------------------
# _parse_ast
# ---------------------------------------------------------------------------


class TestParseAst:
    def test_basic_stats(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text(
            textwrap.dedent(
                '''\
                """Module docstring."""

                def foo():
                    pass

                class Bar:
                    def method(self):
                        pass
                '''
            )
        )
        funcs, classes, has_doc, imports = _parse_ast(f)
        assert funcs == 2  # foo + method
        assert classes == 1
        assert has_doc is True
        assert imports == []

    def test_no_docstring(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("x = 1\n")
        _, _, has_doc, _ = _parse_ast(f)
        assert has_doc is False

    def test_alloc_imports(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text(
            textwrap.dedent(
                """\
                from alloc.config.settings import get_settings
                import alloc.lib.cache
                """
            )
        )
        _, _, _, imports = _parse_ast(f)
        assert "alloc.config.settings" in imports
        assert "alloc.lib.cache" in imports

    def test_non_alloc_imports_ignored(self, tmp_path: Path) -> None:
        f = tmp_path / "mod.py"
        f.write_text("import os\nfrom pathlib import Path\n")
        _, _, _, imports = _parse_ast(f)
        assert imports == []

    def test_syntax_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.py"
        f.write_text("def foo(\n")
        funcs, classes, has_doc, imports = _parse_ast(f)
        assert funcs == 0
        assert classes == 0
        assert has_doc is False
        assert imports == []


# ---------------------------------------------------------------------------
# _find_test_count
# ---------------------------------------------------------------------------


class TestFindTestCount:
    def test_matches_test_file(self, tmp_path: Path) -> None:
        pkg = tmp_path / "fakepkg"
        pkg.mkdir()
        mod = pkg / "core.py"
        mod.write_text("x = 1\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_core.py").write_text(
            textwrap.dedent(
                """\
                def test_one():
                    pass

                def test_two():
                    pass

                def helper():
                    pass
                """
            )
        )
        count = _find_test_count(mod, tests, pkg)
        assert count == 2

    def test_no_matching_test_file(self, tmp_path: Path) -> None:
        pkg = tmp_path / "fakepkg"
        pkg.mkdir()
        mod = pkg / "core.py"
        mod.write_text("x = 1\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        count = _find_test_count(mod, tests, pkg)
        assert count == 0

    def test_nested_module_test(self, tmp_path: Path) -> None:
        pkg = tmp_path / "fakepkg"
        pkg.mkdir()
        lib = pkg / "lib"
        lib.mkdir(parents=True, exist_ok=True)
        mod = lib / "cache.py"
        mod.write_text("x = 1\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_cache.py").write_text(
            textwrap.dedent(
                """\
                def test_cache_hit():
                    pass
                """
            )
        )
        count = _find_test_count(mod, tests, pkg)
        assert count == 1


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


class TestSignalDetection:
    def test_s1_no_tests(self) -> None:
        stats = ModuleStats(
            path="alloc.foo", lines=10, functions=1,
            classes=0, has_docstring=True, test_count=0,
        )
        assert _detect_s1_no_tests(stats) is True

    def test_s1_has_tests(self) -> None:
        stats = ModuleStats(
            path="alloc.foo", lines=10, functions=1,
            classes=0, has_docstring=True, test_count=3,
        )
        assert _detect_s1_no_tests(stats) is False

    def test_s2_oversized(self) -> None:
        stats = ModuleStats(
            path="alloc.foo", lines=250, functions=20,
            classes=2, has_docstring=True, test_count=1,
        )
        assert _detect_s2_oversized(stats) is True

    def test_s2_not_oversized_lines(self) -> None:
        stats = ModuleStats(
            path="alloc.foo", lines=100, functions=20,
            classes=2, has_docstring=True, test_count=1,
        )
        assert _detect_s2_oversized(stats) is False

    def test_s2_not_oversized_funcs(self) -> None:
        stats = ModuleStats(
            path="alloc.foo", lines=250, functions=10,
            classes=2, has_docstring=True, test_count=1,
        )
        assert _detect_s2_oversized(stats) is False

    def test_s3_dead_code(self) -> None:
        stats = ModuleStats(
            path="alloc.orphan", lines=5, functions=1,
            classes=0, has_docstring=True, test_count=1,
        )
        all_imports = {
            "alloc.core": ["alloc.lib.cache"],
            "alloc.orphan": [],
        }
        assert _detect_s3_dead_code(stats, all_imports) is True

    def test_s3_not_dead(self) -> None:
        stats = ModuleStats(
            path="alloc.lib.cache", lines=5, functions=1,
            classes=0, has_docstring=True, test_count=1,
        )
        all_imports = {
            "alloc.core": ["alloc.lib.cache"],
            "alloc.lib.cache": [],
        }
        assert _detect_s3_dead_code(stats, all_imports) is False

    def test_s3_submodule_import(self) -> None:
        stats = ModuleStats(
            path="alloc.models", lines=5, functions=1,
            classes=0, has_docstring=True, test_count=1,
        )
        all_imports = {
            "alloc.core": ["alloc.models.networks"],
            "alloc.models": [],
        }
        assert _detect_s3_dead_code(stats, all_imports) is False


# ---------------------------------------------------------------------------
# crawl_package
# ---------------------------------------------------------------------------


class TestCrawlPackage:
    def test_crawl_minimal_package(self, tmp_pkg: Path, tmp_tests: Path) -> None:
        # Add a module with a test
        (tmp_pkg / "core.py").write_text(
            textwrap.dedent(
                '''\
                """Core module."""

                def run():
                    pass
                '''
            )
        )
        (tmp_tests / "test_core.py").write_text(
            textwrap.dedent(
                """\
                def test_run():
                    pass
                """
            )
        )

        meta = crawl_package(tmp_pkg, tmp_tests)
        assert meta.total_modules == 2  # __init__ + core
        assert meta.total_functions >= 1
        assert meta.total_tests >= 1

    def test_crawl_nonexistent_package(self) -> None:
        with pytest.raises(FileNotFoundError):
            crawl_package("/nonexistent/package")

    def test_crawl_detects_s1(self, tmp_pkg: Path, tmp_tests: Path) -> None:
        # Module with no test
        (tmp_pkg / "orphan.py").write_text(
            textwrap.dedent(
                '''\
                """Orphan module."""

                def orphan_func():
                    pass
                '''
            )
        )
        meta = crawl_package(tmp_pkg, tmp_tests)
        orphan_mod = [
            m for m in meta.modules if m["path"].endswith("orphan")
        ]
        assert len(orphan_mod) == 1
        assert "S1:no_tests" in orphan_mod[0]["signals"]

    def test_crawl_detects_s2(self, tmp_pkg: Path, tmp_tests: Path) -> None:
        # Oversized module: >200 lines AND >15 functions
        lines = '"""Big module."""\n\n'
        for i in range(25):
            # Each function has 10 lines of body to exceed 200 total
            body = "\n".join(f"    x{i}_{j} = {j}" for j in range(10))
            lines += f"def func_{i}():\n{body}\n\n"
        (tmp_pkg / "big.py").write_text(lines)
        (tmp_tests / "test_big.py").write_text("def test_big(): pass\n")

        meta = crawl_package(tmp_pkg, tmp_tests)
        big_mod = [m for m in meta.modules if m["path"].endswith("big")]
        assert len(big_mod) == 1
        assert big_mod[0]["lines"] > 200
        assert big_mod[0]["functions"] > 15
        assert "S2:oversized" in big_mod[0]["signals"]


# ---------------------------------------------------------------------------
# generate_json
# ---------------------------------------------------------------------------


class TestGenerateJson:
    def test_returns_valid_json(self, tmp_pkg: Path, tmp_tests: Path) -> None:
        (tmp_pkg / "core.py").write_text('"""Core."""\n\ndef run(): pass\n')
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)
        assert "package" in data
        assert "modules" in data
        assert "total_modules" in data

    def test_writes_to_file(self, tmp_pkg: Path, tmp_tests: Path,
                            tmp_path: Path) -> None:
        (tmp_pkg / "core.py").write_text('"""Core."""\n\ndef run(): pass\n')
        out = tmp_path / "dashboard.json"
        generate_json(tmp_pkg, tmp_tests, output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["total_modules"] >= 1

    def test_json_structure(self, tmp_pkg: Path, tmp_tests: Path) -> None:
        (tmp_pkg / "core.py").write_text(
            textwrap.dedent(
                '''\
                """Core."""
                from alloc.lib.cache import DiskCache

                def run():
                    pass
                '''
            )
        )
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)
        # Check module entry structure
        core_mod = [m for m in data["modules"] if m["path"].endswith("core")]
        assert len(core_mod) == 1
        assert "lines" in core_mod[0]
        assert "functions" in core_mod[0]
        assert "classes" in core_mod[0]
        assert "test_count" in core_mod[0]
        assert "signals" in core_mod[0]
        assert "has_docstring" in core_mod[0]
        assert "imports" in core_mod[0]
