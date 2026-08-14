"""Tests for alloc.lib.cycle_signals — health signal tree viewer."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from alloc.lib.cycle_signals import (
    SIGNAL_LABELS,
    _signal_label,
    render_tree,
    load_and_render,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metadata() -> dict:
    """Return a minimal but valid metadata dict for testing."""
    return {
        "package": "alloc",
        "total_modules": 3,
        "total_lines": 500,
        "total_functions": 40,
        "total_classes": 5,
        "total_tests": 12,
        "signals_summary": {"S1": 1, "S3": 1},
        "modules": [
            {
                "path": "alloc.__init__",
                "lines": 5,
                "functions": 0,
                "classes": 0,
                "has_docstring": True,
                "imports": [],
                "test_count": 0,
                "signals": ["S1:no_tests", "S3:dead_code"],
            },
            {
                "path": "alloc.core",
                "lines": 300,
                "functions": 20,
                "classes": 3,
                "has_docstring": True,
                "imports": ["alloc.lib.cache"],
                "test_count": 8,
                "signals": ["S2:oversized"],
            },
            {
                "path": "alloc.lib.cache",
                "lines": 195,
                "functions": 10,
                "classes": 1,
                "has_docstring": True,
                "imports": ["alloc.config.settings"],
                "test_count": 4,
                "signals": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# _signal_label
# ---------------------------------------------------------------------------


class TestSignalLabel:
    def test_s1_label(self) -> None:
        assert "no tests" in _signal_label("S1:no_tests")

    def test_s2_label(self) -> None:
        assert "oversized" in _signal_label("S2:oversized")

    def test_s3_label(self) -> None:
        assert "dead code" in _signal_label("S3:dead_code")

    def test_s4_label(self) -> None:
        assert "lint errors" in _signal_label("S4:lint_errors(3)")

    def test_unknown_signal(self) -> None:
        result = _signal_label("X9:unknown")
        assert "X9:unknown" in result


# ---------------------------------------------------------------------------
# SIGNAL_LABELS
# ---------------------------------------------------------------------------


class TestSignalLabels:
    def test_all_keys_present(self) -> None:
        assert "S1" in SIGNAL_LABELS
        assert "S2" in SIGNAL_LABELS
        assert "S3" in SIGNAL_LABELS
        assert "S4" in SIGNAL_LABELS


# ---------------------------------------------------------------------------
# render_tree
# ---------------------------------------------------------------------------


class TestRenderTree:
    def test_returns_string(self, sample_metadata: dict) -> None:
        result = render_tree(sample_metadata)
        assert isinstance(result, str)

    def test_contains_package_name(self, sample_metadata: dict) -> None:
        result = render_tree(sample_metadata)
        assert "alloc" in result

    def test_contains_module_paths(self, sample_metadata: dict) -> None:
        result = render_tree(sample_metadata)
        assert "alloc.core" in result
        assert "alloc.lib.cache" in result

    def test_contains_signal_info(self, sample_metadata: dict) -> None:
        result = render_tree(sample_metadata)
        assert "S1" in result
        assert "S3" in result

    def test_contains_stats(self, sample_metadata: dict) -> None:
        result = render_tree(sample_metadata)
        assert "500" in result  # total lines
        assert "40" in result   # total functions

    def test_no_signals(self) -> None:
        meta = {
            "package": "clean_pkg",
            "total_modules": 1,
            "total_lines": 10,
            "total_functions": 1,
            "total_classes": 0,
            "total_tests": 1,
            "signals_summary": {},
            "modules": [
                {
                    "path": "clean_pkg.core",
                    "lines": 10,
                    "functions": 1,
                    "classes": 0,
                    "has_docstring": True,
                    "imports": [],
                    "test_count": 1,
                    "signals": [],
                },
            ],
        }
        result = render_tree(meta)
        assert "all clear" in result

    def test_empty_modules(self) -> None:
        meta = {
            "package": "empty_pkg",
            "total_modules": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_tests": 0,
            "signals_summary": {},
            "modules": [],
        }
        result = render_tree(meta)
        assert "no modules found" in result

    def test_badges_shown(self, sample_metadata: dict) -> None:
        result = render_tree(sample_metadata)
        # __init__ has no tests and is dead code
        assert "no-tests" in result
        # core is oversized
        assert "300L" in result
        assert "20F" in result

    def test_tree_structure_characters(self, sample_metadata: dict) -> None:
        result = render_tree(sample_metadata)
        assert "├──" in result or "└──" in result
        assert "┌─ Modules" in result


# ---------------------------------------------------------------------------
# load_and_render
# ---------------------------------------------------------------------------


class TestLoadAndRender:
    def test_load_from_file(self, tmp_path: Path, sample_metadata: dict) -> None:
        json_file = tmp_path / "dashboard.json"
        json_file.write_text(json.dumps(sample_metadata))
        result = load_and_render(json_file)
        assert isinstance(result, str)
        assert "alloc" in result

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_render("/nonexistent/dashboard.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all")
        with pytest.raises(json.JSONDecodeError):
            load_and_render(bad_file)
