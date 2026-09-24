# TidyBot Robosuite simulator service

This package is the Robosuite peer of `robocasa_sim`. It owns the official
Robosuite environment, task/controller lifecycle, reset/seed behavior,
camera/proprioception filtering, native success evaluator, and server-side
reference diagnostics.

```bash
python -m robosuite_sim --host 127.0.0.1 --port 8082
```

Control API:

- `GET /health`
- `GET /v1/capabilities`
- `POST /v1/reset`
- `POST /v1/step`
- `GET /v1/observation`
- `GET /v1/success`
- `GET /v1/metadata`
- `POST /v1/reference`
- `POST /v1/close`

The service does not contain an agent, benchmark protocol, model client, or
artifact writer. Those remain in AttentionHarness. Privileged object state and
the native evaluator never appear in policy observations.

The default service is the frozen v1 non-oracle boundary. To run the separate
v2 GT-perception development track, start a dedicated instance with
`python -m robosuite_sim --enable-sim-gt --host 127.0.0.1 --port 8082`.
Only this opt-in instance exposes `POST /v2/perceive_gt` and reset-variation
attestation. Never use that instance for v1 non-oracle scores; restrict its
network access to trusted lab processes.

Public camera observations use flat array keys:

- `<camera>_image`: uint8 RGB with OpenCV top-left origin
- `<camera>_depth`: metric depth in meters with the same top-left origin
- `<camera>_intrinsics`: 3x3 pinhole matrix
- `<camera>_pose_mat`: 4x4 OpenCV-camera-to-Robosuite-world transform

The v1 capability document declares `object_oracle_visible=false`; v1 policy
observations remain filtered. The opt-in v2 endpoint deliberately returns
simulator object poses for GT-labelled runs only. Rewards and evaluator
internals remain outside the shared SDK.
