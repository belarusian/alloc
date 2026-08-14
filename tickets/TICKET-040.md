# TICKET-040: Add integration test for full dashboard pipeline (crawl → JSON → HTML → sync)

**Module:** `tests/test_publish_dashboard.py`, `alloc/lib/publish_dashboard.py`
**Priority:** Medium — end-to-end validation
**Cycle:** 21

## What's Wrong

The test suite for `publish_dashboard` covers unit tests (helpers, HTML
generation, file writing) but lacks an integration test that runs the full
pipeline: crawl a real package → generate JSON → render HTML → verify the
output contains expected module data.

## Evidence

- `tests/test_publish_dashboard.py` — all tests use hand-crafted `sample_metadata`
  dicts; none call `generate_html()` with real output from `dashboard.crawl_package()`
- `tests/test_dashboard.py` — tests `crawl_package` and `generate_json` but never
  pipes the result into `publish_dashboard`
- No test exercises the `main()` CLI entry point of `publish_dashboard`

## Impact

If the JSON schema from `dashboard.crawl_package()` changes (e.g. a field is
renamed or nested differently), `publish_dashboard` could silently produce
broken HTML without any test catching it. The contract between the two modules
is unverified at integration level.

## Suggestion

Add an integration test class:
