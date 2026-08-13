# TICKET-008: alloc/lib/client.py — Add get_last_trade to Cache Map

**Module:** `alloc/lib/client.py`
**Test:** `tests/test_client.py` (add test case)
**Priority:** Medium — real-time prediction mode makes uncached API calls per ticker.

## Evidence

`alloc/lib/client.py` line 30-33 defines the `CACHE_MAP`:
