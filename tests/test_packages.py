"""Verify all alloc packages are importable and versioned."""

import importlib


def test_alloc_importable():
    mod = importlib.import_module("alloc")
    assert mod is not None


def test_alloc_version_non_empty():
    import alloc
    assert alloc.__version__
    assert len(alloc.__version__.strip()) > 0


def test_alloc_lib_importable():
    mod = importlib.import_module("alloc.lib")
    assert mod is not None


def test_alloc_config_importable():
    mod = importlib.import_module("alloc.config")
    assert mod is not None


def test_alloc_models_importable():
    mod = importlib.import_module("alloc.models")
    assert mod is not None
