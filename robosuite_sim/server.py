"""Thread-safe HTTP control service for the Robosuite backend."""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

from .backend import BackendConfig, RobosuiteBackend
from .codec import decode_array, encode_array, encode_observation


BackendFactory = Callable[[BackendConfig], RobosuiteBackend]


class SimulatorState:
    def __init__(
        self, backend_factory: BackendFactory = RobosuiteBackend,
        *, enable_sim_gt: bool = False,
    ) -> None:
        self._backend_factory = backend_factory
        self.enable_sim_gt = enable_sim_gt
        self._backend: RobosuiteBackend | None = None
        self._config: BackendConfig | None = None
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "service": "robosuite_sim",
                "active": self._backend is not None,
                "task_id": self._config.task_id if self._config else None,
            }

    def capabilities(self) -> dict[str, Any]:
        """Describe the stable, non-privileged Robot Service boundary."""

        return {
            "service": "robosuite_sim",
            "api_version": "v1",
            "control_frame": "robosuite_world",
            "policy_operations": ["attach", "observe", "step"],
            "harness_operations": ["reset", "success", "reference", "close"],
            "sensors": ["rgb", "metric_depth", "camera_intrinsics", "camera_pose", "proprioception"],
            "object_oracle_visible": False,
        }

    def reset(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.enable_sim_gt and (
            "variation" in request or "discover_variation" in request
        ):
            raise PermissionError("sim_gt variation attestation is disabled on this service")
        config = BackendConfig(
            task_id=str(request["task_id"]),
            horizon=int(request.get("horizon", 500)),
            camera=bool(request.get("camera", True)),
            camera_name=str(request.get("camera_name", "agentview")),
            camera_height=int(request.get("camera_height", 256)),
            camera_width=int(request.get("camera_width", 256)),
        )
        seed = int(request["seed"])
        with self._lock:
            if self._backend is None or self._config != config:
                self._close_unlocked()
                self._backend = self._backend_factory(config)
                self._config = config
            observation = self._backend.reset(seed)
            requested = request.get("variation")
            discovery = request.get("discover_variation") is True
            applied = None
            if requested is not None or discovery:
                applied = self._backend.variation_identity()
                if requested is not None and requested != applied:
                    raise ValueError("reset did not realize requested scene/object variation")
            low, high = self._backend.action_spec
            result = {
                "observation": encode_observation(observation),
                "action_low": encode_array(low),
                "action_high": encode_array(high),
                "metadata": self._backend.metadata,
            }
            if applied is not None:
                result["applied_variation"] = applied
            return result

    def perceive_gt(self, request: dict[str, Any]) -> dict[str, Any]:
        """Explicit v2 sim_gt endpoint; never part of the frozen v1 SDK."""
        if not self.enable_sim_gt:
            raise PermissionError("sim_gt perception is disabled on this service")
        with self._lock:
            backend = self._require_backend()
            requested_cameras = request.get("camera_names")
            selected = [backend.config.camera_name]
            if requested_cameras is not None and requested_cameras != selected:
                raise ValueError("GT perception camera does not match the active camera")
            target_names = request.get("target_names")
            if target_names is not None and (
                not isinstance(target_names, list)
                or any(not isinstance(name, str) for name in target_names)
            ):
                raise ValueError("target_names must be a list of strings")
            objects = backend.gt_objects()
            if target_names is not None:
                objects = [item for item in objects if item["name"] in target_names]
            return {"objects": objects, "cameras": selected, "frame": "robosuite_world"}

    def step(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            backend = self._require_backend()
            observation, reward, done, info = backend.step(decode_array(request["action"]))
            return {
                "observation": encode_observation(observation),
                "reward": reward,
                "done": done,
                "info": info,
            }

    def observe(self) -> dict[str, Any]:
        with self._lock:
            return {"observation": encode_observation(self._require_backend().observe())}

    def session(self) -> dict[str, Any]:
        """Describe the active public session without resetting it.

        This lets a separately sandboxed policy process attach to an episode
        while keeping the simulator and its native evaluator in this service.
        """
        with self._lock:
            backend = self._require_backend()
            low, high = backend.action_spec
            return {
                "observation": encode_observation(backend.observe()),
                "action_low": encode_array(low),
                "action_high": encode_array(high),
                "metadata": backend.metadata,
            }

    def success(self) -> dict[str, Any]:
        with self._lock:
            return {"native_success": self._require_backend().native_success()}

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return self._require_backend().metadata

    def reference(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            observation = self._require_backend().run_reference(
                float(request.get("timeout_seconds", 60.0))
            )
            return {
                "observation": encode_observation(observation),
                "trace": list(self._require_backend().trace),
            }

    def close(self) -> dict[str, Any]:
        with self._lock:
            self._close_unlocked()
            return {"closed": True}

    def _close_unlocked(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._backend = None
        self._config = None

    def _require_backend(self) -> RobosuiteBackend:
        if self._backend is None:
            raise RuntimeError("no active environment; call /v1/reset first")
        return self._backend


def handler_factory(state: SimulatorState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            pass

        def _send(self, status: int, value: dict[str, Any]) -> None:
            payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length)) if length else {}

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            routes = {
                "/health": state.health,
                "/v1/capabilities": state.capabilities,
                "/v1/session": state.session,
                "/v1/observation": state.observe,
                "/v1/success": state.success,
                "/v1/metadata": state.metadata,
            }
            self._dispatch(routes.get(path), {})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            routes = {
                "/v1/reset": state.reset,
                "/v1/step": state.step,
                "/v1/reference": state.reference,
                "/v1/close": lambda _request: state.close(),
                **({"/v2/perceive_gt": state.perceive_gt} if state.enable_sim_gt else {}),
            }
            self._dispatch(routes.get(path), self._body())

        def _dispatch(self, route, request: dict[str, Any]) -> None:
            if route is None:
                self._send(404, {"error": "not found"})
                return
            try:
                value = route(request) if request or self.command == "POST" else route()
                self._send(200, value)
            except Exception as exc:
                self._send(
                    500,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )

    return Handler


def create_server(host: str, port: int, state: SimulatorState | None = None) -> HTTPServer:
    state = state or SimulatorState()
    # MuJoCo's offscreen GL context is thread-affine. A single HTTP server
    # thread keeps reset / step / render calls on the context's owner thread.
    return HTTPServer((host, port), handler_factory(state))
