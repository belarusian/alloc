# TICKET-046: HHI normalization diverges from seed — document + regression-guard

**Status:** OPEN
**Date:** 2025-08-16
**Cycle:** 36
**Priority:** Low
**Issue:** https://github.com/belarusian/alloc/issues/112

## Summary

The seed's `calculate_portfolio_statistics` computes HHI directly over the
non-cash allocation fractions. Because those fractions sum to < 1 whenever
cash is held, the seed's "normalized" HHI
`(hhi - 1/n) / (1 - 1/n)` can go **negative** (observed -0.45 in a 3-asset,
cash-heavy portfolio), which is outside the documented [0, 1] range.

alloc's implementation deliberately renormalizes the non-cash weights to sum
to 1 before squaring, so HHI ∈ [1/n, 1] and the normalized form ∈ [0, 1].
This is an intentional improvement over the seed, but it means alloc's HHI
values are **not numerically equal** to the seed's for the same portfolio.

## Evidence

- Seed `trader/models/portfolio.py` `calculate_portfolio_statistics` (line
  ~648): `hhi = sum(np.square(non_cash_allocations))` where
  `non_cash_allocations` are raw allocation fractions (sum < 1 with cash).
- alloc `alloc/models/portfolio.py` `calculate_portfolio_statistics`:
  `weights = [w / total_nc for w in non_cash]` then `hhi = sum(square(weights))`.
- Reproduced: 3 equal non-cash positions + large cash → seed normalized HHI
  ≈ -0.45; alloc normalized HHI ≈ 0.0.

## Impact

- Any downstream consumer expecting seed-identical HHI numbers will see a
  difference. This is a parity gap by design, not a bug, but it must be
  documented so it is not "fixed" back into the negative range.

## Suggestion (implementation plan)

1. Add a module-level note in `alloc/models/portfolio.py` (or docs/) stating
   that HHI is computed on renormalized non-cash weights and therefore
   differs from the seed's raw-fraction HHI.
2. Add a regression test asserting `hhi_normalized` stays within [0, 1] for a
   cash-heavy portfolio (guard against reintroducing the negative range).
3. If strict seed parity is ever required, add an optional
   `renormalize: bool = True` parameter to
   `calculate_portfolio_statistics` and document the trade-off.

## Acceptance criteria

- Documented divergence between alloc and seed HHI semantics.
- Regression test pins `hhi_normalized ∈ [0, 1]` for cash-heavy portfolios.
