"""Official-Robosuite environment lifecycle owned by the simulator service."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .tasks import TaskSpec, get_task


EXPECTED_VERSIONS = {
    "robosuite": "1.5.1",
    "mujoco": "3.3.0",
    "numpy": "1.26.4",
}


class DependencyVersionError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendConfig:
    task_id: str
    horizon: int = 500
    camera: bool = True
    camera_name: str = "agentview"
    camera_height: int = 256
    camera_width: int = 256
    gl_backend: str = "egl"


def verify_runtime_versions() -> dict[str, str]:
    actual = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    mismatches = [
        f"{name}=={actual[name]} (expected {expected})"
        for name, expected in EXPECTED_VERSIONS.items()
        if actual[name] != expected
    ]
    if mismatches:
        raise DependencyVersionError("incompatible simulator versions: " + ", ".join(mismatches))
    return actual


class RobosuiteBackend:
    """One active Robosuite environment; never imported by AttentionHarness."""

    def __init__(self, config: BackendConfig) -> None:
        os.environ.setdefault("MUJOCO_GL", config.gl_backend)
        versions = verify_runtime_versions()

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        module_path = Path(suite.__file__).resolve()
        if "ASPIRE" in module_path.parts or "aspire" in module_path.parts:
            raise RuntimeError(f"refusing non-independent Robosuite import: {module_path}")

        self.config = config
        self.spec: TaskSpec = get_task(config.task_id)
        self._versions = versions
        self._module_path = module_path
        self._seed: int | None = None
        self._last_raw: Mapping[str, Any] | None = None
        self.trace: list[dict[str, Any]] = []
        self._env = suite.make(
            env_name=self.spec.robosuite_env,
            robots="Panda",
            controller_configs=load_composite_controller_config(robot="Panda"),
            has_renderer=False,
            has_offscreen_renderer=config.camera,
            use_camera_obs=config.camera,
            use_object_obs=True,
            camera_names=config.camera_name,
            camera_heights=config.camera_height,
            camera_widths=config.camera_width,
            camera_depths=config.camera,
            horizon=config.horizon,
            ignore_done=True,
            hard_reset=True,
            reward_shaping=False,
        )

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        low, high = self._env.action_spec
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "service": "robosuite_sim",
            "service_api_version": "v1",
            "backend_class": f"{type(self._env).__module__}.{type(self._env).__name__}",
            "task_id": self.spec.task_id,
            "robosuite_env": self.spec.robosuite_env,
            "seed": self._seed,
            "control_frame": "robosuite_world",
            "observation_schema": "tidybot.robot-observation.v1",
            "robosuite_origin": str(self._module_path),
            "versions": self._versions,
        }

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        caller_rng_state = np.random.get_state()
        try:
            np.random.seed(seed)
            raw = self._env.reset()
        finally:
            np.random.set_state(caller_rng_state)
        self._seed = seed
        self._last_raw = raw
        self.trace.clear()
        return self._public_observation(raw, phase="reset")

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        low, high = self.action_spec
        array = np.asarray(action, dtype=np.float64)
        if array.shape != low.shape:
            raise ValueError(f"action shape {array.shape} does not match {low.shape}")
        if not np.isfinite(array).all():
            raise ValueError("action must contain only finite values")
        applied_action = np.clip(array, low, high)
        raw, reward, done, info = self._env.step(applied_action)
        self._last_raw = raw
        # Record the action result before any camera conversion or rerender.
        entry = {"step": len(self.trace), "action": applied_action.tolist(),
                 "reward": float(reward), "done": bool(done), "action_executed": True}
        self.trace.append(entry)
        hook = getattr(self, "_receipt_hook", None)
        if hook is not None:
            hook(entry)
        depth_recovery = None
        try:
            public = self._public_observation(raw, phase="step", action=applied_action)
        except ValueError as exc:
            prefix = "invalid normalized depth frame: "
            if not str(exc).startswith(prefix):
                raise
            diagnostics = json.loads(str(exc)[len(prefix):])
            try:
                repeat, repeat_raw = self._probe_depth_repeat()
            except Exception as probe_error:
                repeat, repeat_raw = {"status": "probe_error", "observation_only": True,
                    "simulation_time_unchanged": False, "qpos_unchanged": False,
                    "error": f"{type(probe_error).__name__}: {probe_error}"}, None
            entry["depth_recovery"] = diagnostics
            diagnostics["repeat_observation"] = repeat
            recoverable = (
                repeat_raw is not None
                and repeat.get("status") == "valid"
                and repeat.get("simulation_time_unchanged") is True
                and repeat.get("qpos_unchanged") is True
                and "capture_sha256" in diagnostics
                and "capture_sha256" in repeat
                and repeat["intrinsics_before"] == repeat["intrinsics_after"]
                and repeat["pose_before"] == repeat["pose_after"]
            )
            if not recoverable:
                raise ValueError(prefix + json.dumps(diagnostics, sort_keys=True)) from exc
            try:
                public = self._public_observation(
                    repeat_raw, phase="step_repeat", action=applied_action)
                equivalence = self._recovery_equivalence(raw, repeat_raw, public)
                diagnostics["equivalence"] = equivalence
                if equivalence["passed"] is not True:
                    raise ValueError("recovery observation is not equivalent")
            except Exception as recovery_error:
                diagnostics["recovery_error"] = f"{type(recovery_error).__name__}: {recovery_error}"
                raise ValueError(prefix + json.dumps(diagnostics, sort_keys=True)) from exc
            self._last_raw = repeat_raw
            depth_recovery = diagnostics
        public_info = dict(info)
        if depth_recovery is not None:
            public_info["depth_recovered_without_physics_step"] = True
            repeat = depth_recovery["repeat_observation"]
            public_info["depth_recovery"] = {
                "first_frame_sha256": depth_recovery["frame_sha256"],
                "first_capture_sha256": depth_recovery["capture_sha256"],
                "first_capture_file": Path(depth_recovery["capture_path"]).name,
                "repeat_frame_sha256": repeat["frame_sha256"],
                "repeat_capture_sha256": repeat["capture_sha256"],
                "repeat_capture_file": Path(repeat["capture_path"]).name,
                "simulation_time_unchanged": repeat["simulation_time_unchanged"],
                "qpos_unchanged": repeat["qpos_unchanged"],
                "observation_only": repeat["observation_only"],
            }
        return public, float(reward), bool(done), public_info

    def observe(self) -> dict[str, np.ndarray]:
        if self._last_raw is None:
            raise RuntimeError("reset must be called before observe")
        return self._public_observation(self._last_raw, phase="observe")

    def native_success(self) -> bool:
        return bool(self._env._check_success())

    def gt_objects(self) -> list[dict[str, Any]]:
        """Evaluator-side object positions for the separate v2 sim_gt track."""
        if self._last_raw is None:
            raise RuntimeError("reset must be called before GT perception")
        objects = []
        for key in self.spec.object_keys:
            position = np.asarray(self._last_raw[key], dtype=np.float64).reshape(-1)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"invalid task object position: {key}")
            objects.append({"name": key.removesuffix("_pos"), "position": position.tolist()})
        return objects

    def variation_identity(self) -> dict[str, str]:
        """Identify the realized reset, not merely the requested seed."""
        if self._last_raw is None:
            raise RuntimeError("reset must be called before variation attestation")
        objects = [
            [item["name"], *[round(value, 4) for value in item["position"]]]
            for item in self.gt_objects()
        ]
        eef = np.asarray(self._last_raw["robot0_eef_pos"], dtype=np.float64).reshape(-1)
        if eef.shape != (3,) or not np.isfinite(eef).all():
            raise ValueError("invalid robot start position")
        def digest(value: Any) -> str:
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]
        return {
            "scene_id": "scene:" + digest({"objects": objects, "robot_start": [round(float(x), 4) for x in eef]}),
            "object_set_id": "objects:" + digest(objects),
        }

    def run_reference(self, timeout_seconds: float) -> dict[str, np.ndarray]:
        from .reference_policy import run_reference_policy

        run_reference_policy(self, timeout_seconds)
        return self.observe()

    def reference_observation(self) -> Mapping[str, Any]:
        return self._env._get_observations()

    def close(self) -> None:
        self._env.close()

    def _public_observation(
        self, raw: Mapping[str, Any], *, phase: str = "observe",
        action: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        public = self._filter_observation(raw)
        if not self.config.camera:
            return public

        camera = self.config.camera_name
        image_key = f"{camera}_image"
        depth_key = f"{camera}_depth"
        # Robosuite's observation arrays retain MuJoCo's bottom-left image
        # origin, while the published pinhole intrinsics and corrected camera
        # pose use the OpenCV top-left convention. Canonicalize both RGB and
        # depth here so every public camera field shares one pixel coordinate
        # system.
        if image_key in public:
            public[image_key] = np.flipud(public[image_key]).copy()
        if depth_key in public:
            try:
                metric_depth = self._metric_depth(self._env.sim, np.flipud(public[depth_key]))
            except ValueError as exc:
                prefix = "invalid normalized depth frame: "
                if not str(exc).startswith(prefix):
                    raise
                diagnostics = json.loads(str(exc)[len(prefix):])
                diagnostics["observation_phase"] = phase
                diagnostics["camera_name"] = camera
                diagnostics["backend_step_index"] = max(0, len(self.trace) - 1)
                if action is not None:
                    diagnostics["action_sha256"] = hashlib.sha256(
                        np.ascontiguousarray(action).tobytes()).hexdigest()
                raise ValueError(prefix + json.dumps(diagnostics, sort_keys=True)) from exc
            public[depth_key] = np.asarray(metric_depth, dtype=np.float32)
        public[f"{camera}_intrinsics"] = np.asarray(
            self._camera_intrinsics(
                self._env.sim,
                camera,
                self.config.camera_height,
                self.config.camera_width,
            ),
            dtype=np.float64,
        )
        public[f"{camera}_pose_mat"] = np.asarray(
            self._camera_pose(self._env.sim, camera),
            dtype=np.float64,
        )
        return public

    def _probe_depth_repeat(self) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
        """Sample depth once more without a physics step after an invalid frame."""
        sim = self._env.sim
        data = getattr(sim, "data", None)
        before_time = getattr(data, "time", None)
        before_qpos = np.asarray(data.qpos).copy() if hasattr(data, "qpos") else None
        before_intrinsics = self._camera_intrinsics(sim, self.config.camera_name,
            self.config.camera_height, self.config.camera_width)
        before_pose = self._camera_pose(sim, self.config.camera_name)
        result: dict[str, Any] = {"observation_only": True}
        raw: Mapping[str, Any] | None = None
        try:
            raw = self._env._get_observations(force_update=True)
            key = f"{self.config.camera_name}_depth"
            if key not in raw:
                result["status"] = "depth_missing"
            else:
                frame = np.ascontiguousarray(np.flipud(raw[key]))
                normalized = np.asarray(frame, dtype=np.float64)
                finite = normalized[np.isfinite(normalized)]
                result.update({
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                    "finite_min": float(finite.min()) if finite.size else None,
                    "finite_max": float(finite.max()) if finite.size else None,
                    "nonfinite_count": int(normalized.size - finite.size),
                })
                try:
                    self._metric_depth(sim, frame)
                except ValueError as exc:
                    prefix = "invalid normalized depth frame: "
                    if not str(exc).startswith(prefix):
                        raise
                    result["status"] = "invalid"
                    result["invalid_diagnostics"] = json.loads(str(exc)[len(prefix):])
                    for key in ("capture_path", "capture_sha256", "capture_error"):
                        if key in result["invalid_diagnostics"]:
                            result[key] = result["invalid_diagnostics"][key]
                else:
                    result["status"] = "valid"
                capture_dir = os.environ.get("TIDYBOT_INVALID_DEPTH_DIR")
                if capture_dir and "capture_path" not in result:
                    try:
                        directory = Path(capture_dir).resolve()
                        directory.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(
                            dir=directory, prefix=f"repeat-{result['frame_sha256']}-",
                            suffix=".npy", delete=False,
                        ) as tmp:
                            np.save(tmp, frame, allow_pickle=False)
                            tmp.flush()
                            os.fsync(tmp.fileno())
                            target = Path(tmp.name)
                        result["capture_path"] = str(target)
                        result["capture_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                    except Exception as exc:
                        result["capture_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            result["status"] = "probe_error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            after_time = getattr(data, "time", None)
            result["simulation_time_before"] = before_time
            result["simulation_time_after"] = after_time
            result["qpos_before"] = before_qpos.tolist() if before_qpos is not None else None
            result["qpos_after"] = np.asarray(data.qpos).tolist() if hasattr(data, "qpos") else None
            result["intrinsics_before"] = before_intrinsics.tolist()
            result["intrinsics_after"] = self._camera_intrinsics(sim, self.config.camera_name,
                self.config.camera_height, self.config.camera_width).tolist()
            result["pose_before"] = before_pose.tolist()
            result["pose_after"] = self._camera_pose(sim, self.config.camera_name).tolist()
            result["simulation_time_unchanged"] = (
                before_time is not None and before_time == after_time)
            if before_qpos is not None:
                result["qpos_unchanged"] = bool(np.array_equal(before_qpos, data.qpos))
            else:
                result["qpos_unchanged"] = False
        return result, raw

    def _recovery_equivalence(self, original, repeated, public):
        camera = self.config.camera_name
        filtered = self._filter_observation(original)
        expected_keys = set(filtered) | {f"{camera}_intrinsics", f"{camera}_pose_mat"}
        checks = {"same_keys": set(public) == expected_keys,
                  "same_dtype": True, "same_shape": True,
                  "proprioception_unchanged": True}
        for key, value in filtered.items():
            expected_dtype = np.dtype("float32") if key.endswith("_depth") else value.dtype
            checks["same_dtype"] &= key in public and public[key].dtype == expected_dtype
            checks["same_shape"] &= key in public and public[key].shape == value.shape
            if key.startswith("robot0_"):
                checks["proprioception_unchanged"] &= key in public and np.array_equal(value, public[key])
        depth = public[f"{camera}_depth"]
        normalized = np.flipud(repeated[f"{camera}_depth"])
        sim = self._env.sim
        near = float(sim.model.vis.map.znear) * float(sim.model.stat.extent)
        far = float(sim.model.vis.map.zfar) * float(sim.model.stat.extent)
        expected = (near / (1 - np.asarray(normalized, dtype=np.float64) * (1 - near / far))).astype(np.float32)
        checks["legal_metric_depth"] = bool(np.isfinite(depth).all() and (depth > 0).all()
            and np.array_equal(depth, expected))
        # Capture complete public fields and source normalized depth for independent replay.
        directory = Path(os.environ["TIDYBOT_INVALID_DEPTH_DIR"]).resolve()
        with tempfile.NamedTemporaryFile(dir=directory, prefix="equivalence-", suffix=".npz", delete=False) as stream:
            np.savez(stream, **{"original__" + k: v for k, v in filtered.items()},
                     **{"public__" + k: v for k, v in public.items()}, normalized_depth=normalized,
                     near=np.array(near), far=np.array(far))
            stream.flush()
            os.fsync(stream.fileno())
            path = Path(stream.name)
        return {"checks": {k: bool(v) for k, v in checks.items()},
                "passed": all(checks.values()), "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    @staticmethod
    def _metric_depth(sim: Any, depth: np.ndarray) -> np.ndarray:
        """Convert MuJoCo's normalized depth buffer to meters."""

        normalized = np.asarray(depth, dtype=np.float64)
        invalid = ~np.isfinite(normalized) | (normalized < 0.0) | (normalized > 1.0)
        if np.any(invalid):
            finite = normalized[np.isfinite(normalized)]
            frame = np.ascontiguousarray(depth)
            diagnostics = {
                "shape": list(normalized.shape),
                "source_dtype": str(np.asarray(depth).dtype),
                "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                "finite_count": int(finite.size),
                "nonfinite_count": int(normalized.size - finite.size),
                "below_zero_count": int(np.count_nonzero(normalized < 0.0)),
                "above_one_count": int(np.count_nonzero(normalized > 1.0)),
                "finite_min": float(finite.min()) if finite.size else None,
                "finite_max": float(finite.max()) if finite.size else None,
                "first_invalid_index": list(map(int, np.argwhere(invalid)[0])),
            }
            first = normalized[tuple(diagnostics["first_invalid_index"])]
            diagnostics["first_invalid_kind"] = (
                "nan" if np.isnan(first) else
                "positive_infinity" if np.isposinf(first) else
                "negative_infinity" if np.isneginf(first) else
                "below_zero" if first < 0.0 else "above_one"
            )
            capture_dir = os.environ.get("TIDYBOT_INVALID_DEPTH_DIR")
            if capture_dir:
                try:
                    directory = Path(capture_dir).resolve()
                    directory.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        dir=directory, prefix=f"{diagnostics['frame_sha256']}-",
                        suffix=".npy", delete=False,
                    ) as tmp:
                        target = Path(tmp.name)
                        np.save(tmp, frame, allow_pickle=False)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                    diagnostics["capture_path"] = str(target)
                    diagnostics["capture_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                except Exception as exc:
                    diagnostics["capture_error"] = f"{type(exc).__name__}: {exc}"
            raise ValueError("invalid normalized depth frame: " + json.dumps(diagnostics, sort_keys=True))
        extent = float(sim.model.stat.extent)
        far = float(sim.model.vis.map.zfar) * extent
        near = float(sim.model.vis.map.znear) * extent
        return near / (1.0 - normalized * (1.0 - near / far))

    @staticmethod
    def _camera_intrinsics(
        sim: Any,
        camera_name: str,
        camera_height: int,
        camera_width: int,
    ) -> np.ndarray:
        camera_id = sim.model.camera_name2id(camera_name)
        fovy = float(sim.model.cam_fovy[camera_id])
        focal = 0.5 * camera_height / np.tan(fovy * np.pi / 360.0)
        return np.array(
            [
                [focal, 0.0, camera_width / 2.0],
                [0.0, focal, camera_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _camera_pose(sim: Any, camera_name: str) -> np.ndarray:
        """Return OpenCV camera coordinates transformed into simulator world."""

        camera_id = sim.model.camera_name2id(camera_name)
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = np.asarray(
            sim.data.cam_xmat[camera_id], dtype=np.float64
        ).reshape(3, 3)
        camera_to_world[:3, 3] = np.asarray(
            sim.data.cam_xpos[camera_id], dtype=np.float64
        )
        axis_correction = np.diag([1.0, -1.0, -1.0, 1.0])
        return camera_to_world @ axis_correction

    @staticmethod
    def _filter_observation(raw: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Strip privileged task/object state before adding public calibration."""

        public: dict[str, np.ndarray] = {}
        for key, value in raw.items():
            if key.endswith("_image") or key.endswith("_depth") or key.startswith("robot0_"):
                public[key] = np.array(value, copy=True)
        return public
