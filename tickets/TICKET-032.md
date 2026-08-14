# TICKET-032: Add unit test for `ActorCriticNetworks._soft_update_targets()`

## What's Wrong

`_soft_update_targets()` (line 286–308 in `alloc/models/networks.py`) has **zero test coverage**. This method implements the polyak averaging that stabilizes DDPG target networks:
