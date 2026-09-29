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

Depth containment (1.1.0): every step requires a reset/attach session ID and a
request ID. The client creates an ID per action; a failed call exposes
`ServiceError.request_id` and `session_id`. Retry with that same ID (and action)
to replay the durable response, including HTTP 500 receipts. A new ID is a new
action. Unresolved intents are never re-executed, including after restart.
Set `TIDYBOT_RECEIPT_DIR` to the attempt's retained evidence directory. The
service fsyncs intent, then reward/done/applied action immediately after physics,
then the observation or structured stop response before returning HTTP.

One bounded observation-only rerender may recover a normalized-depth error.
Both frames and a public-observation equivalence NPZ are retained, including
simulation time, qpos and camera calibration before and after rerender. Any
missing evidence, changed state/schema/calibration/proprioception, or invalid
metric depth stops the environment. Recovery requires independent evidence
review before comparison admission. Historical bad-frame root cause remains
unknown; bounded validation does not prove future errors impossible.
