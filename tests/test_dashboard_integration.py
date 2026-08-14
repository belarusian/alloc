"""Integration tests for the full dashboard pipeline.

Tests the end-to-end flow: crawl → JSON → HTML → publish → sync.
Also tests the CLI --publish-dashboard flag wiring.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typing import Any

from alloc.cli import build_parser, parse_args
from alloc.lib.dashboard import crawl_package, generate_json
from alloc.lib.publish_dashboard import generate_html, publish


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_pkg(tmp_path: Path) -> Path:
    """Create a minimal fake package under tmp_path."""
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""fake package."""\n')
    (pkg / "core.py").write_text(
        textwrap.dedent(
            '''\
            """Core module."""

            def run():
                pass

            class Runner:
                def execute(self):
                    pass
            '''
        )
    )
    (pkg / "utils.py").write_text(
        textwrap.dedent(
            '''\
            """Utilities module."""
            from fakepkg.core import run

            def helper():
                return run()
            '''
        )
    )
    return pkg


@pytest.fixture
def tmp_tests(tmp_path: Path) -> Path:
    """Create a minimal tests directory under tmp_path."""
    td = tmp_path / "tests"
    td.mkdir()
    (td / "test_core.py").write_text(
        textwrap.dedent(
            """\
            def test_run():
                pass

            def test_runner():
                pass
            """
        )
    )
    return td


# ---------------------------------------------------------------------------
# Full pipeline: crawl → JSON → HTML → publish
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end dashboard pipeline tests."""

    def test_crawl_to_json(self, tmp_pkg: Path, tmp_tests: Path) -> None:
        """crawl_package produces valid DashboardMetadata."""
        meta = crawl_package(tmp_pkg, tmp_tests)
        assert meta.total_modules >= 3  # __init__, core, utils
        assert meta.total_lines > 0
        assert meta.total_functions > 0
        assert meta.total_tests >= 2  # test_core has 2 tests

    def test_crawl_to_json_to_html(
        self, tmp_pkg: Path, tmp_tests: Path
    ) -> None:
        """Full crawl → JSON → HTML pipeline produces valid HTML."""
        meta = crawl_package(tmp_pkg, tmp_tests)
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)

        html = generate_html(data)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<style>" in html
        assert "<script>" in html
        assert "viewport" in html
        assert "width=device-width" in html

    def test_pipeline_writes_html_file(
        self, tmp_pkg: Path, tmp_tests: Path, tmp_path: Path
    ) -> None:
        """Full pipeline writes a valid HTML file to disk."""
        meta = crawl_package(tmp_pkg, tmp_tests)
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)

        html = generate_html(data)
        out = tmp_path / "dashboard.html"
        result = publish(html, output_path=out, sync=False)

        assert result == out
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "fakepkg" in content

    def test_pipeline_detects_signals(
        self, tmp_pkg: Path, tmp_tests: Path
    ) -> None:
        """Pipeline correctly detects S1 (no tests) on utils module."""
        meta = crawl_package(tmp_pkg, tmp_tests)
        utils_mod = [m for m in meta.modules if m["path"].endswith("utils")]
        assert len(utils_mod) == 1
        # utils has no test file → S1
        assert "S1:no_tests" in utils_mod[0]["signals"]

    def test_pipeline_json_roundtrip(
        self, tmp_pkg: Path, tmp_tests: Path
    ) -> None:
        """JSON output can be re-loaded and used to generate HTML."""
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)

        # Verify structure
        assert "package" in data
        assert "modules" in data
        assert "signals_summary" in data
        assert "total_modules" in data

        # Generate HTML from loaded JSON
        html = generate_html(data)
        assert "<!DOCTYPE html>" in html

    def test_pipeline_with_all_signal_types(
        self, tmp_pkg: Path, tmp_tests: Path
    ) -> None:
        """Pipeline handles modules triggering multiple signal types."""
        # Create an oversized module (>200 lines, >15 functions)
        lines = '"""Big module."""\n\n'
        for i in range(25):
            body = "\n".join(f"    x{i}_{j} = {j}" for j in range(8))
            lines += f"def func_{i}():\n{body}\n\n"
        (tmp_pkg / "big.py").write_text(lines)

        meta = crawl_package(tmp_pkg, tmp_tests)
        big_mod = [m for m in meta.modules if m["path"].endswith("big")]
        assert len(big_mod) == 1
        assert "S2:oversized" in big_mod[0]["signals"]
        assert "S1:no_tests" in big_mod[0]["signals"]

        # Generate HTML and verify signal badges appear
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)
        html = generate_html(data)
        assert "badge-s1" in html
        assert "badge-s2" in html


# ---------------------------------------------------------------------------
# GitHub Pages sync tests
# ---------------------------------------------------------------------------


class TestSyncPipeline:
    """Tests for the --sync flag in the publish pipeline."""

    def test_sync_not_called_by_default(
        self, tmp_pkg: Path, tmp_tests: Path, tmp_path: Path
    ) -> None:
        """publish() does not call _sync_to_ghpages when sync=False."""
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)
        html = generate_html(data)
        out = tmp_path / "dashboard.html"

        with patch(
            "alloc.lib.publish_dashboard._sync_to_ghpages"
        ) as mock_sync:
            publish(html, output_path=out, sync=False)
            mock_sync.assert_not_called()

    def test_sync_called_when_flagged(
        self, tmp_pkg: Path, tmp_tests: Path, tmp_path: Path
    ) -> None:
        """publish() calls _sync_to_ghpages when sync=True."""
        json_str = generate_json(tmp_pkg, tmp_tests)
        data = json.loads(json_str)
        html = generate_html(data)
        out = tmp_path / "dashboard.html"

        with patch(
            "alloc.lib.publish_dashboard._sync_to_ghpages"
        ) as mock_sync:
            publish(html, output_path=out, sync=True)
            mock_sync.assert_called_once_with(out)


# ---------------------------------------------------------------------------
# CLI --publish-dashboard flag tests
# ---------------------------------------------------------------------------


class TestCLIPublishDashboard:
    """Tests for --publish-dashboard CLI flag wiring."""

    def test_parser_has_publish_dashboard_flag(self) -> None:
        """build_parser includes --publish-dashboard argument."""
        parser = build_parser()
        # Parse with --publish-dashboard should not raise
        args = parser.parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 50000}',
            "--publish-dashboard",
        ])
        assert args.publish_dashboard is True

    def test_parser_has_dashboard_output_flag(self) -> None:
        """build_parser includes --dashboard-output argument."""
        parser = build_parser()
        args = parser.parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 50000}',
            "--dashboard-output", "custom.html",
        ])
        assert args.dashboard_output == "custom.html"

    def test_parser_has_dashboard_sync_flag(self) -> None:
        """build_parser includes --dashboard-sync argument."""
        parser = build_parser()
        args = parser.parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 50000}',
            "--dashboard-sync",
        ])
        assert args.dashboard_sync is True

    def test_publish_dashboard_defaults_false(self) -> None:
        """--publish-dashboard defaults to False."""
        args = parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 50000}',
        ])
        assert args.publish_dashboard is False

    def test_dashboard_output_defaults_dashboard_html(self) -> None:
        """--dashboard-output defaults to 'dashboard.html'."""
        args = parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 50000}',
        ])
        assert args.dashboard_output == "dashboard.html"

    def test_dashboard_sync_defaults_false(self) -> None:
        """--dashboard-sync defaults to False."""
        args = parse_args([
            "--tickers", "AAPL",
            "--positions-values", '{"AAPL": 50000}',
        ])
        assert args.dashboard_sync is False

    def test_main_generates_dashboard_when_flagged(self, caplog: Any) -> None:
        """main() generates dashboard when --publish-dashboard is set."""
        import logging
        from types import SimpleNamespace

        caplog.set_level(logging.INFO)

        mock_meta = SimpleNamespace(
            package="alloc",
            total_modules=1,
            total_lines=100,
            total_functions=10,
            total_classes=1,
            total_tests=5,
            modules=[],
            signals_summary={},
        )

        with patch("alloc.core.create_trainer") as mock_create, \
             patch("alloc.cli.WorkflowRunner") as mock_runner, \
             patch(
                 "alloc.lib.dashboard.crawl_package",
                 return_value=mock_meta,
             ) as mock_crawl, \
             patch(
                 "alloc.lib.publish_dashboard.generate_html",
                 return_value="<html></html>",
             ) as mock_gen, \
             patch(
                 "alloc.lib.publish_dashboard.publish",
                 return_value=Path("dashboard.html"),
             ) as mock_pub:

            # Setup mocks
            mock_trainer = MagicMock()
            mock_create.return_value = mock_trainer

            mock_result = MagicMock()
            mock_result.status = "success"
            mock_result.trials = []
            mock_result.best_trial = MagicMock(
                iteration=0, allocation=[], recommended_trades=None
            )
            mock_result.concentration = None
            mock_runner.return_value.run.return_value = mock_result

            from alloc.cli import main
            code = main([
                "--tickers", "AAPL",
                "--positions-values", '{"AAPL": 50000}',
                "--publish-dashboard",
            ])

            assert code == 0
            mock_crawl.assert_called_once()
            mock_gen.assert_called_once()
            mock_pub.assert_called_once()

    def test_main_skips_dashboard_when_not_flagged(
        self, caplog: Any
    ) -> None:
        """main() does not generate dashboard without --publish-dashboard."""
        import logging
        caplog.set_level(logging.INFO)

        with patch("alloc.core.create_trainer") as mock_create, \
             patch("alloc.cli.WorkflowRunner") as mock_runner:

            mock_trainer = MagicMock()
            mock_create.return_value = mock_trainer

            mock_result = MagicMock()
            mock_result.status = "success"
            mock_result.trials = []
            mock_result.best_trial = MagicMock(
                iteration=0, allocation=[], recommended_trades=None
            )
            mock_result.concentration = None
            mock_runner.return_value.run.return_value = mock_result

            from alloc.cli import main
            code = main([
                "--tickers", "AAPL",
                "--positions-values", '{"AAPL": 50000}',
            ])

            assert code == 0
            # crawl_package should NOT have been called
            assert "Generating health dashboard" not in caplog.text


# ---------------------------------------------------------------------------
# Responsive design tests
# ---------------------------------------------------------------------------


class TestResponsiveDesign:
    """Tests for responsive HTML output."""

    def test_html_has_viewport_meta(self) -> None:
        """Generated HTML includes viewport meta tag for mobile."""
        data = {
            "package": "test",
            "total_modules": 1,
            "total_lines": 10,
            "total_functions": 1,
            "total_classes": 0,
            "total_tests": 1,
            "signals_summary": {},
            "modules": [],
        }
        html = generate_html(data)
        assert 'name="viewport"' in html
        assert "width=device-width" in html
        assert "initial-scale=1.0" in html

    def test_html_has_media_queries(self) -> None:
        """Generated HTML includes @media queries for responsive layout."""
        data = {
            "package": "test",
            "total_modules": 1,
            "total_lines": 10,
            "total_functions": 1,
            "total_classes": 0,
            "total_tests": 1,
            "signals_summary": {},
            "modules": [],
        }
        html = generate_html(data)
        assert "@media" in html

    def test_html_has_flexbox_layout(self) -> None:
        """Generated HTML uses flexbox for responsive grid."""
        data = {
            "package": "test",
            "total_modules": 1,
            "total_lines": 10,
            "total_functions": 1,
            "total_classes": 0,
            "total_tests": 1,
            "signals_summary": {},
            "modules": [],
        }
        html = generate_html(data)
        assert "flex-wrap" in html or "flex" in html

    def test_html_is_standalone(self) -> None:
        """Generated HTML is fully standalone (no external CSS/JS deps)."""
        data = {
            "package": "test",
            "total_modules": 1,
            "total_lines": 10,
            "total_functions": 1,
            "total_classes": 0,
            "total_tests": 1,
            "signals_summary": {},
            "modules": [],
        }
        html = generate_html(data)
        # Should have inline <style> and <script>
        assert "<style>" in html
        assert "<script>" in html
        # Should NOT reference external CSS/JS files
        assert 'href="http' not in html
        assert 'src="http' not in html
