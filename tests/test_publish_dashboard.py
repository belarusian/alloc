"""Tests for alloc.lib.publish_dashboard — HTML dashboard publisher."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from alloc.lib.publish_dashboard import (
    SIGNAL_COLORS,
    SIGNAL_ICONS,
    SIGNAL_LABELS,
    _escape_html,
    _signal_badge,
    _module_row,
    generate_html,
    publish,
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
# Signal constants
# ---------------------------------------------------------------------------


class TestSignalConstants:
    def test_all_signal_colors_present(self) -> None:
        for key in ("S1", "S2", "S3", "S4"):
            assert key in SIGNAL_COLORS
            bg, fg = SIGNAL_COLORS[key]
            assert isinstance(bg, str) and bg.startswith("#")
            assert isinstance(fg, str) and fg.startswith("#")

    def test_all_signal_icons_present(self) -> None:
        for key in ("S1", "S2", "S3", "S4"):
            assert key in SIGNAL_ICONS
            assert len(SIGNAL_ICONS[key]) > 0

    def test_all_signal_labels_present(self) -> None:
        for key in ("S1", "S2", "S3", "S4"):
            assert key in SIGNAL_LABELS
            assert len(SIGNAL_LABELS[key]) > 0


# ---------------------------------------------------------------------------
# _escape_html
# ---------------------------------------------------------------------------


class TestEscapeHtml:
    def test_escapes_angle_brackets(self) -> None:
        assert _escape_html("<script>") == "&lt;script&gt;"

    def test_escapes_ampersand(self) -> None:
        assert _escape_html("a & b") == "a &amp; b"

    def test_escapes_quotes(self) -> None:
        assert _escape_html('say "hi"') == "say &quot;hi&quot;"

    def test_plain_text_unchanged(self) -> None:
        assert _escape_html("hello world") == "hello world"


# ---------------------------------------------------------------------------
# _signal_badge
# ---------------------------------------------------------------------------


class TestSignalBadge:
    def test_s1_badge(self) -> None:
        html = _signal_badge("S1:no_tests")
        assert "badge-s1" in html
        assert "no_tests" in html

    def test_s4_badge_with_count(self) -> None:
        html = _signal_badge("S4:lint_errors(3)")
        assert "badge-s4" in html
        assert "lint_errors(3)" in html

    def test_badge_has_inline_style(self) -> None:
        html = _signal_badge("S2:oversized")
        assert "background:" in html
        assert "color:" in html


# ---------------------------------------------------------------------------
# _module_row
# ---------------------------------------------------------------------------


class TestModuleRow:
    def test_returns_tr_element(self, sample_metadata: dict) -> None:
        row = _module_row(sample_metadata["modules"][0])
        assert row.startswith("<tr>")
        assert row.endswith("</tr>")

    def test_contains_module_path(self, sample_metadata: dict) -> None:
        row = _module_row(sample_metadata["modules"][0])
        assert "alloc.__init__" in row

    def test_clear_module_has_clear_badge(self, sample_metadata: dict) -> None:
        row = _module_row(sample_metadata["modules"][2])  # no signals
        assert "clear" in row


# ---------------------------------------------------------------------------
# generate_html
# ---------------------------------------------------------------------------


class TestGenerateHtml:
    def test_returns_string(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_is_valid_html_structure(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "</head>" in html
        assert "<body>" in html
        assert "</body>" in html

    def test_contains_package_name(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert "alloc" in html

    def test_contains_summary_values(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert "500" in html  # total lines
        assert "40" in html   # total functions
        assert "12" in html   # total tests

    def test_contains_signal_badges(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert "badge-s1" in html
        assert "badge-s3" in html

    def test_contains_inline_css(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert "<style>" in html
        assert "responsive" in html.lower() or "@media" in html

    def test_contains_inline_js(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert "<script>" in html
        assert "filterTable" in html

    def test_contains_viewport_meta(self, sample_metadata: dict) -> None:
        html = generate_html(sample_metadata)
        assert 'viewport' in html
        assert 'width=device-width' in html

    def test_empty_metadata(self) -> None:
        html = generate_html({})
        assert "<!DOCTYPE html>" in html
        assert "unknown" in html

    def test_module_with_xss_attempt(self) -> None:
        metadata = {
            "package": "test",
            "total_modules": 1,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_tests": 0,
            "signals_summary": {},
            "modules": [
                {
                    "path": '<script>alert("xss")</script>',
                    "lines": 0,
                    "functions": 0,
                    "classes": 0,
                    "has_docstring": False,
                    "imports": [],
                    "test_count": 0,
                    "signals": [],
                }
            ],
        }
        html = generate_html(metadata)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


class TestPublish:
    def test_writes_file(self, sample_metadata: dict, tmp_path: Path) -> None:
        html = generate_html(sample_metadata)
        out = tmp_path / "out.html"
        result = publish(html, output_path=out)
        assert result == out
        assert out.exists()
        assert "<!DOCTYPE html>" in out.read_text()

    def test_creates_parent_dirs(self, sample_metadata: dict, tmp_path: Path) -> None:
        html = generate_html(sample_metadata)
        out = tmp_path / "nested" / "deep" / "dashboard.html"
        publish(html, output_path=out)
        assert out.exists()

    def test_no_sync_by_default(
        self, sample_metadata: dict, tmp_path: Path
    ) -> None:
        html = generate_html(sample_metadata)
        out = tmp_path / "dashboard.html"
        with patch(
            "alloc.lib.publish_dashboard._sync_to_ghpages"
        ) as mock_sync:
            publish(html, output_path=out, sync=False)
            mock_sync.assert_not_called()

    def test_sync_when_flagged(
        self, sample_metadata: dict, tmp_path: Path
    ) -> None:
        html = generate_html(sample_metadata)
        out = tmp_path / "dashboard.html"
        with patch(
            "alloc.lib.publish_dashboard._sync_to_ghpages"
        ) as mock_sync:
            publish(html, output_path=out, sync=True)
            mock_sync.assert_called_once()
