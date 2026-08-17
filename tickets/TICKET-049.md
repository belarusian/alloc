# TICKET-049: Document HHI renormalization divergence from seed + pin hhi_normalized ∈ [0,1] regression test

**Status:** OPEN
**Date:** 2025-08-16
**Cycle:** 39
**Priority:** Low
**Related:** TICKET-046 (analysis of the same divergence)

## Summary

`alloc/models/portfolio.py::calculate_portfolio_statistics` computes HHI on
**renormalized** non-cash weights (divided by their sum so they total 1.0),
whereas the seed `trader/models/portfolio.py` computes HHI on the **raw**
non-cash allocation fractions (which sum to < 1 whenever cash is held).

This is an intentional improvement: the seed's raw-fraction form makes the
"normalized" HHI `(hhi - 1/n)/(1 - 1/n)` go **negative** for cash-heavy
portfolios (observed ≈ -0.45 for a 3-asset, large-cash portfolio), which is
outside the documented [0, 1] range. The alloc renormalized form keeps HHI ∈
[1/n, 1] and the normalized form ∈ [0, 1].

The divergence is currently only described in a short inline comment
(lines 338–341) and is **not** documented at module level, and there is **no
regression test** pinning `hhi_normalized` to [0, 1] for a cash-heavy
portfolio.

## Evidence

- `alloc/models/portfolio.py` lines 342–355:
