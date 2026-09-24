# TidyBot Robosuite Sim Service

An independently installable HTTP service for the official Robosuite environment.
It owns reset, stepping, public observation filtering, and the native success
evaluator. It does not import AttentionHarness, ASPIRE, an agent, or a model.

```bash
pip install -e .
python -m robosuite_sim --host 127.0.0.1 --port 8082
```

The service and its `RobosuiteSimClient` use the versioned `/v1` contract
described in `robosuite_sim/README.md`. `GET /health` reports readiness.
Keep evaluator endpoints on the harness side of the policy sandbox; they are
not policy observations. Run the contract smoke test with:

```bash
python -m pytest tests/test_service_api.py
```

The initial source is the exact `robosuite_sim/` snapshot in the frozen
TidyBot AttentionBench v1 protocol. That snapshot remains in Tidybot-Universe
for historical hash verification; new integrations should pin this service
repository to a specific commit rather than copying service code.
